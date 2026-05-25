import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod

import httpx

from app.config import Settings
from app.utils.errors import ExternalProviderError

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalEmbeddingClient(EmbeddingClient):
    """Deterministic vectorizer for local development when no API key is configured."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features: list[str] = []
        for token in tokens:
            features.append(token)
            features.extend(token[i : i + 3] for i in range(max(len(token) - 2, 0)))

        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1 if digest[4] % 2 == 0 else -1
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ExternalProviderError("OPENAI_API_KEY is required for OpenAI embeddings.", 401)
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.settings.embedding_model, "input": texts}
        if self.settings.embedding_dimensions:
            payload["dimensions"] = self.settings.embedding_dimensions
        data = await _post_json(
            "https://api.openai.com/v1/embeddings",
            payload,
            {"Authorization": f"Bearer {self.settings.openai_api_key}"},
            self.settings.embedding_timeout_seconds,
        )
        usage = data.get("usage")
        if usage:
            logger.info("OpenAI embedding token usage: %s", usage)
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ExternalProviderError("GEMINI_API_KEY is required for Gemini embeddings.", 401)
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        try:
            async with httpx.AsyncClient(timeout=self.settings.embedding_timeout_seconds) as client:
                for text in texts:
                    url = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{self.settings.embedding_model}:embedContent?key={self.settings.gemini_api_key}"
                    )
                    response = await client.post(
                        url,
                        json={"content": {"parts": [{"text": text}]}},
                    )
                    _raise_for_provider_status(response)
                    embeddings.append(response.json()["embedding"]["values"])
        except httpx.TimeoutException as exc:
            raise ExternalProviderError("Embedding provider request timed out.", 504) from exc
        except httpx.RequestError as exc:
            raise ExternalProviderError("Embedding provider request failed.", 503) from exc
        return embeddings


class MistralEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings):
        if not settings.mistral_api_key:
            raise ExternalProviderError("MISTRAL_API_KEY is required for Mistral embeddings.", 401)
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await _post_json(
            "https://api.mistral.ai/v1/embeddings",
            {"model": self.settings.embedding_model, "input": texts},
            {"Authorization": f"Bearer {self.settings.mistral_api_key}"},
            self.settings.embedding_timeout_seconds,
        )
        usage = data.get("usage")
        if usage:
            logger.info("Mistral embedding token usage: %s", usage)
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingClient(settings)
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingClient(settings)
    if settings.embedding_provider == "mistral":
        return MistralEmbeddingClient(settings)
    return LocalEmbeddingClient(settings.embedding_dimensions)


async def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            _raise_for_provider_status(response)
            return response.json()
    except httpx.TimeoutException as exc:
        raise ExternalProviderError("Embedding provider request timed out.", 504) from exc
    except httpx.RequestError as exc:
        raise ExternalProviderError("Embedding provider request failed.", 503) from exc


def _raise_for_provider_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code in {401, 403}:
        raise ExternalProviderError("Embedding provider rejected the API key.", 401)
    if response.status_code == 429:
        raise ExternalProviderError("Embedding provider rate limit exceeded.", 429)
    raise ExternalProviderError(f"Embedding provider returned HTTP {response.status_code}.", 502)
