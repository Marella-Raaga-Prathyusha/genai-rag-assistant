import logging
import re
from abc import ABC, abstractmethod

import httpx

from app.config import Settings
from app.utils.errors import ExternalProviderError

logger = logging.getLogger(__name__)


FALLBACK_ANSWER = "I could not find enough information in the knowledge base to answer this question."


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: str) -> str:
        raise NotImplementedError


class LocalLLMClient(LLMClient):
    """Grounded extractive responder for local demos without provider credentials."""

    async def generate(self, prompt: str, context: str) -> str:
        del prompt
        clean_context = "\n".join(
            line for line in context.splitlines() if not line.startswith("[")
        )
        sentences = re.split(r"(?<=[.!?])\s+", clean_context.strip())
        useful = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]
        if not useful:
            return FALLBACK_ANSWER
        return " ".join(useful[:3])


class OpenAILLMClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ExternalProviderError("OPENAI_API_KEY is required for OpenAI chat.", 401)
        self.settings = settings

    async def generate(self, prompt: str, context: str) -> str:
        del context
        data = await _post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {self.settings.openai_api_key}"},
            self.settings.llm_timeout_seconds,
            "LLM",
        )
        _log_usage("OpenAI chat", data.get("usage"))
        return data["choices"][0]["message"]["content"].strip()


class GeminiLLMClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ExternalProviderError("GEMINI_API_KEY is required for Gemini chat.", 401)
        self.settings = settings

    async def generate(self, prompt: str, context: str) -> str:
        del context
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.llm_model}:generateContent?key={self.settings.gemini_api_key}"
        )
        data = await _post_json(
            url,
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.settings.llm_temperature},
            },
            {},
            self.settings.llm_timeout_seconds,
            "LLM",
        )
        usage = data.get("usageMetadata")
        _log_usage("Gemini chat", usage)
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


class AnthropicLLMClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise ExternalProviderError("ANTHROPIC_API_KEY is required for Claude chat.", 401)
        self.settings = settings

    async def generate(self, prompt: str, context: str) -> str:
        del context
        data = await _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.settings.llm_model,
                "max_tokens": 800,
                "temperature": self.settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            self.settings.llm_timeout_seconds,
            "LLM",
        )
        _log_usage("Claude chat", data.get("usage"))
        return "".join(block.get("text", "") for block in data["content"]).strip()


class MistralLLMClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.mistral_api_key:
            raise ExternalProviderError("MISTRAL_API_KEY is required for Mistral chat.", 401)
        self.settings = settings

    async def generate(self, prompt: str, context: str) -> str:
        del context
        data = await _post_json(
            "https://api.mistral.ai/v1/chat/completions",
            {
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {self.settings.mistral_api_key}"},
            self.settings.llm_timeout_seconds,
            "LLM",
        )
        _log_usage("Mistral chat", data.get("usage"))
        return data["choices"][0]["message"]["content"].strip()


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAILLMClient(settings)
    if settings.llm_provider == "gemini":
        return GeminiLLMClient(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicLLMClient(settings)
    if settings.llm_provider == "mistral":
        return MistralLLMClient(settings)
    return LocalLLMClient()


async def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_seconds: float,
    label: str,
) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code in {401, 403}:
                raise ExternalProviderError(f"{label} provider rejected the API key.", 401)
            if response.status_code == 429:
                raise ExternalProviderError(f"{label} provider rate limit exceeded.", 429)
            if response.status_code >= 400:
                raise ExternalProviderError(f"{label} provider returned HTTP {response.status_code}.", 502)
            return response.json()
    except httpx.TimeoutException as exc:
        raise ExternalProviderError(f"{label} provider request timed out.", 504) from exc
    except httpx.RequestError as exc:
        raise ExternalProviderError(f"{label} provider request failed.", 503) from exc


def _log_usage(label: str, usage: dict | None) -> None:
    if usage:
        logger.info("%s token usage: %s", label, usage)
