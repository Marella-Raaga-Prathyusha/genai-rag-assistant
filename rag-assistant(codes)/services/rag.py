import logging
from datetime import datetime, timezone

from app.config import Settings
from app.models.schemas import ChatResponse, SourceChunk
from app.prompts.templates import build_prompt
from app.services.embeddings import EmbeddingClient
from app.services.llm import FALLBACK_ANSWER, LLMClient
from app.vectorstore.sqlite_store import SQLiteVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteVectorStore,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
    ):
        self.settings = settings
        self.store = store
        self.embedding_client = embedding_client
        self.llm_client = llm_client

    async def answer(self, session_id: str, message: str) -> ChatResponse:
        history_rows = self.store.get_recent_pairs(session_id, self.settings.conversation_pairs)
        history = _format_history(history_rows)

        query_embedding = (await self.embedding_client.embed([message]))[0]
        results = self.store.search(query_embedding, self.settings.top_k)
        logger.info(
            "Similarity scores session=%s scores=%s",
            session_id,
            [round(result.similarity, 4) for result in results],
        )

        above_threshold = [
            result for result in results if result.similarity >= self.settings.similarity_threshold
        ]
        sources = [
            SourceChunk(
                title=result.title,
                chunk_id=result.chunk_id,
                source_document=result.source_document,
                similarity=result.similarity,
                text=result.text,
            )
            for result in above_threshold
        ]

        self.store.add_chat_message(session_id, "user", message)
        if not above_threshold:
            self.store.add_chat_message(session_id, "assistant", FALLBACK_ANSWER)
            return ChatResponse(
                sessionId=session_id,
                answer=FALLBACK_ANSWER,
                sources=[],
                fallback=True,
                createdAt=datetime.now(timezone.utc),
            )

        retrieved_context = "\n\n".join(
            f"[{result.title} | {result.chunk_id} | {result.source_document}]\n{result.text}"
            for result in above_threshold
        )
        prompt = build_prompt(retrieved_context, history, message)
        answer = await self.llm_client.generate(prompt, retrieved_context)
        if not answer:
            answer = FALLBACK_ANSWER
        self.store.add_chat_message(session_id, "assistant", answer)

        return ChatResponse(
            sessionId=session_id,
            answer=answer,
            sources=sources,
            fallback=answer == FALLBACK_ANSWER,
            createdAt=datetime.now(timezone.utc),
        )


def _format_history(rows) -> str:
    lines = []
    for row in rows:
        role = "User" if row["role"] == "user" else "Assistant"
        lines.append(f"{role}: {row['content']}")
    return "\n".join(lines)
