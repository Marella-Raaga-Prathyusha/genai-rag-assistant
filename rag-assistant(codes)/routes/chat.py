from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import RAGService

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    rag_service: RAGService = request.app.state.rag_service
    return await rag_service.answer(payload.sessionId, payload.message)
