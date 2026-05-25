import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    title: str
    chunk_id: str
    source_document: str
    text: str
    similarity: float


class SQLiteVectorStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_document TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    docs_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, id)")

    def get_metadata(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def replace_chunks(self, chunks: list[dict], docs_hash: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, title, source_document, text, embedding,
                    token_count, docs_hash, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["title"],
                        chunk["source_document"],
                        chunk["text"],
                        json.dumps(chunk["embedding"]),
                        chunk["token_count"],
                        docs_hash,
                        created_at,
                    )
                    for chunk in chunks
                ],
            )
            conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES('docs_hash', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (docs_hash,),
            )

    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(session_id, role, content, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (session_id, role, content, datetime.now(timezone.utc).isoformat()),
            )

    def get_recent_pairs(self, session_id: str, pairs: int) -> list[sqlite3.Row]:
        limit = pairs * 2
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return list(reversed(rows))

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        query_norm = _norm(query_embedding)
        if query_norm == 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT title, chunk_id, source_document, text, embedding FROM chunks"
            ).fetchall()

        scored: list[SearchResult] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            similarity = _cosine_similarity(query_embedding, embedding, query_norm)
            scored.append(
                SearchResult(
                    title=row["title"],
                    chunk_id=row["chunk_id"],
                    source_document=row["source_document"],
                    text=row["text"],
                    similarity=similarity,
                )
            )

        return sorted(scored, key=lambda result: result.similarity, reverse=True)[:top_k]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(left: list[float], right: list[float], left_norm: float | None = None) -> float:
    if len(left) != len(right):
        return 0.0
    left_n = left_norm if left_norm is not None else _norm(left)
    right_n = _norm(right)
    if left_n == 0 or right_n == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_n * right_n)
