from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.services.embeddings import build_embedding_client
from app.services.indexer import ensure_indexed
from app.services.llm import build_llm_client
from app.services.rag import RAGService
from app.utils.errors import (
    ExternalProviderError,
    provider_exception_handler,
    request_validation_exception_handler,
)
from app.utils.logging import configure_logging
from app.vectorstore.sqlite_store import SQLiteVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    store = SQLiteVectorStore(settings.database_path)
    embedding_client = build_embedding_client(settings)
    llm_client = build_llm_client(settings)
    await ensure_indexed(settings, store, embedding_client)
    app.state.rag_service = RAGService(settings, store, embedding_client, llm_client)
    yield


app = FastAPI(
    title="Production RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
    responses={422: {"description": "Validation error"}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_exception_handler(ExternalProviderError, provider_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(frontend_path / "index.html")


app.include_router(health_router)
app.include_router(chat_router)
