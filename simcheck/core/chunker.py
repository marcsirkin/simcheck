"""
Document chunking for semantic analysis.

This module splits documents into chunks suitable for embedding comparison.
The chunking strategy is intentionally simple and opinionated:

Strategy:
1. Split document into sentences
2. Group sentences into chunks of approximately TARGET_TOKENS tokens
3. Never split mid-sentence (semantic boundary preservation)
4. If a single sentence exceeds max size, keep it as its own chunk

Token Estimation:
We use a simple heuristic: ~4 characters per token for English text.
This avoids the overhead of loading a tokenizer and is accurate enough
for chunking purposes. The actual embedding model handles precise tokenization.

Design Decisions:
- No overlap between chunks (simpler analysis, cleaner boundaries)
- Sentence-boundary preservation (maintains semantic coherence)
- Fixed defaults (opinionated, not configurable in v1)
"""

import re
from typing import List
from simcheck.core.models import Chunk


# Default chunk size in approximate tokens.
# 150 tokens (~600 chars) provides granular analysis for diagnostic purposes.
# Smaller chunks make it easier to identify exactly where content drifts
# off-topic, which is the primary use case for this tool.
# Most sentence-transformer models handle 512 tokens max.
DEFAULT_CHUNK_TOKENS = 150

# Approximate characters per token for English text.
# This is a rough heuristic. Actual tokenization varies by model,
# but for chunking purposes this is sufficient.
CHARS_PER_TOKEN = 4


class ChunkingError(Exception):
    """Raised when document chunking fails."""
    pass


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using character-based heuristic.

    This is intentionally approximate. For English text, ~4 chars/token
    is reasonable. The embedding model will handle actual tokenization.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.

    This handles common sentence boundaries: . ! ?
    It attempts to avoid splitting on abbreviations like "Dr." or "U.S."
    by requiring whitespace or end-of-string after the punctuation.

    Args:
        text: Input text

    Returns:
        List of sentence strings (may include trailing whitespace)
    """
    # Pattern: split after .!? followed by whitespace or end
    # This is imperfect but handles most cases without NLP dependencies
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)

    # Filter out empty strings
    return [s for s in sentences if s.strip()]


def chunk_document(
    document: str,
    target_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> List[Chunk]:
    """
    Split a document into semantic chunks for embedding comparison.

    The document is split into sentences, then sentences are grouped
    into chunks of approximately target_tokens size. Sentence boundaries
    are always preserved to maintain semantic coherence.

    Args:
        document: The document text to chunk
        target_tokens: Target size for each chunk in tokens (approximate)

    Returns:
        List of Chunk objects in document order

    Raises:
        ChunkingError: If document is empty or chunking fails
    """
    # Validate input
    if not document or not document.strip():
        raise ChunkingError("Document is empty or contains only whitespace")

    # Normalize whitespace
    document = document.strip()

    # Split into sentences
    sentences = _split_into_sentences(document)

    if not sentences:
        raise ChunkingError("Could not extract any sentences from document")

    # Group sentences into chunks
    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_token_count = 0
    char_position = 0  # Track position in original document

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence would exceed target, finalize current chunk
        # (unless current chunk is empty - then we must include this sentence)
        if current_sentences and (current_token_count + sentence_tokens > target_tokens):
            chunk_text = " ".join(current_sentences)
            chunk_start = char_position
            chunk_end = chunk_start + len(chunk_text)

            chunks.append(Chunk(
                index=len(chunks),
                text=chunk_text,
                char_start=chunk_start,
                char_end=chunk_end,
                token_count=current_token_count,
            ))

            # Move position forward
            char_position = chunk_end
            # Account for the space/newline between chunks in original doc
            # Find where this chunk's text ends in the original document
            while char_position < len(document) and document[char_position] in ' \n\t':
                char_position += 1

            # Start new chunk
            current_sentences = [sentence]
            current_token_count = sentence_tokens
        else:
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_start = char_position
        chunk_end = min(chunk_start + len(chunk_text), len(document))

        chunks.append(Chunk(
            index=len(chunks),
            text=chunk_text,
            char_start=chunk_start,
            char_end=chunk_end,
            token_count=current_token_count,
        ))

    return chunks


def get_total_tokens(chunks: List[Chunk]) -> int:
    """
    Calculate total estimated tokens across all chunks.

    Args:
        chunks: List of Chunk objects

    Returns:
        Sum of token counts
    """
    return sum(chunk.token_count for chunk in chunks)
