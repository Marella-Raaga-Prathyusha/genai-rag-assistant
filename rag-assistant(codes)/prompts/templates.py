SYSTEM_PROMPT = """You are a helpful assistant.

Use ONLY the provided context to answer.

If the context does not contain the answer, say:
"I could not find enough information in the knowledge base to answer this question."

Context:
{retrieved_context}

Conversation History:
{history}

Question:
{user_question}
"""


def build_prompt(retrieved_context: str, history: str, user_question: str) -> str:
    return SYSTEM_PROMPT.format(
        retrieved_context=retrieved_context.strip(),
        history=history.strip() or "No prior conversation.",
        user_question=user_question.strip(),
    )
