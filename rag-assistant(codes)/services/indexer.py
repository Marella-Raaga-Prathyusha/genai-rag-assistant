import hashlib
import json
import logging
from pathlib import Path

from app.config import Settings
from app.services.chunker import chunk_document
from app.services.embeddings import EmbeddingClient
from app.vectorstore.sqlite_store import SQLiteVectorStore

logger = logging.getLogger(__name__)


async def ensure_indexed(
    settings: Settings,
    store: SQLiteVectorStore,
    embedding_client: EmbeddingClient,
) -> None:
    docs_path = Path(settings.docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {docs_path}")

    raw = docs_path.read_text(encoding="utf-8")
    index_config = {
        "chunk_min_tokens": settings.chunk_min_tokens,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
    }
    hash_input = raw + json.dumps(index_config, sort_keys=True)
    docs_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    if store.get_metadata("docs_hash") == docs_hash:
        logger.info("Knowledge base index is up to date.")
        return

    documents = json.loads(raw)
    prepared_chunks = []
    for index, document in enumerate(documents):
        title = document.get("title") or f"Document {index + 1}"
        source_document = document.get("source_document") or document.get("source") or f"doc-{index + 1}"
        content = document.get("content") or document.get("text") or ""
        prepared_chunks.extend(
            chunk_document(
                title=title,
                source_document=source_document,
                content=content,
                min_tokens=settings.chunk_min_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
        )

    if not prepared_chunks:
        logger.warning("No chunks generated from docs.json.")
        store.replace_chunks([], docs_hash)
        return

    texts = [chunk.text for chunk in prepared_chunks]
    embeddings = await embedding_client.embed(texts)
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source_document": chunk.source_document,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "embedding": embedding,
        }
        for chunk, embedding in zip(prepared_chunks, embeddings)
    ]
    store.replace_chunks(rows, docs_hash)
    logger.info("Indexed %s chunks from %s documents.", len(rows), len(documents))
