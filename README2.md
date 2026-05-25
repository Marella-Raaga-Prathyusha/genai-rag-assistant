# GenAI RAG Assistant

A production-style Retrieval-Augmented Generation chat assistant built with FastAPI, Chroma, Gemini, and a lightweight web UI.

This production-ready AI chat application that uses:

* **FastAPI** for the backend APIs
* **SQLite vector database** to store and retrieve document embeddings for RAG
* **Configurable embedding models/providers** like OpenAI or Gemini
* **HTML, CSS, JavaScript frontend** for the chat interface

It retrieves relevant document context using semantic search and sends it to the LLM to generate accurate, grounded responses.


## Features

- FastAPI API with `/api/chat`, `/health`, and session clearing
- Chroma vector retrieval with Gemini embeddings
- Gemini 2.5 Flash generation through `google-generativeai`
- Local fallback retrieval and answering for demos without credentials
- Conversation memory per browser session
- Source snippets and retrieval mode returned with each answer
- Responsive frontend at `/app`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The default `.env.example` uses local development providers so the app runs without external keys. For production, set real providers and API keys:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Gemini, Claude, and Mistral are also wired through `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and `MISTRAL_API_KEY`. Claude is available for chat generation; choose OpenAI, Gemini, or Mistral for embeddings.

## API

### `GET /health`

```json
{
  "status": "healthy"
}
```

### `POST /api/chat`

Request:

```json
{
  "sessionId": "abc123",
  "message": "How can I reset my password?"
}
```

Response:

```json
{
  "sessionId": "abc123",
  "answer": "Open the sign-in page, choose Forgot password...",
  "sources": [
    {
      "title": "Account Access",
      "chunk_id": "support/account-access.md::chunk-0",
      "source_document": "support/account-access.md",
      "similarity": 0.82,
      "text": "..."
    }
  ],
  "fallback": false,
  "createdAt": "2026-05-25T13:30:00Z"
}
```

## RAG Workflow

At startup, the app hashes `docs.json` and compares it with the stored index hash. If the file changed, it reloads documents, chunks text, calls the embedding provider, and replaces the persisted vector index. During chat, retrieval always happens before LLM generation. If no retrieved chunk meets `SIMILARITY_THRESHOLD`, the app returns:

```text
I could not find enough information in the knowledge base to answer this question.
```

The LLM is not called in that fallback path.

## Embedding Strategy

Each chunk stores:

- `title`
- `chunk_id`
- `source_document`
- `text`
- `embedding`
- `token_count`
- index hash metadata

External embedding providers are called in `app/services/embeddings.py`. The local provider is deterministic and intended for development and tests; production deployments should configure an external embedding API.

## Similarity Search

`app/vectorstore/sqlite_store.py` loads persisted vectors and computes cosine similarity:

```text
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
```

Results are sorted by score, capped by `TOP_K`, and filtered by `SIMILARITY_THRESHOLD` before prompt construction.

## Prompt Design

The prompt explicitly instructs the model to use only retrieved context:

```text
You are a helpful assistant.

Use ONLY the provided context to answer.

Context:
{retrieved_context}

Conversation History:
{history}

Question:
{user_question}
```

This keeps the answer grounded and makes the threshold fallback deterministic when retrieval confidence is low.

## Testing

```bash
pytest
```

The tests cover chunk metadata and limits, cosine ranking, and fallback behavior when similarity is below threshold.

