import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    source_document: str
    text: str
    token_count: int


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = text.replace("( ", "(").replace("[ ", "[")
    return re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)


def chunk_document(
    title: str,
    source_document: str,
    content: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    tokens = tokenize(content)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(max_tokens - overlap_tokens, 1)

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]

        if chunks and len(window) < min_tokens and len(chunks[-1].text) > 0:
            previous = chunks.pop()
            merged_tokens = tokenize(previous.text) + window
            chunks.append(
                Chunk(
                    chunk_id=previous.chunk_id,
                    title=title,
                    source_document=source_document,
                    text=detokenize(merged_tokens),
                    token_count=len(merged_tokens),
                )
            )
            break

        chunks.append(
            Chunk(
                chunk_id=f"{source_document}::chunk-{index}",
                title=title,
                source_document=source_document,
                text=detokenize(window),
                token_count=len(window),
            )
        )
        index += 1
        start += step

    return chunks
