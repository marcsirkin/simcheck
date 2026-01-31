"""
Data models for the semantic comparison engine.

These dataclasses define the structured return types used throughout
the comparison pipeline. They are intentionally simple and transparent.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy.typing import NDArray


# Type alias for embedding vectors
Vector = NDArray[np.float32]


@dataclass
class Chunk:
    """
    A segment of document text with metadata.

    Attributes:
        index: Position of this chunk in the original document (0-indexed)
        text: The actual text content of the chunk
        char_start: Starting character position in original document
        char_end: Ending character position in original document
        token_count: Approximate number of tokens in this chunk
    """
    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk."""
        return len(self.text)


@dataclass
class ChunkSimilarity:
    """
    Similarity result for a single chunk compared against a query.

    Attributes:
        chunk: The chunk that was compared
        similarity: Cosine similarity score (0.0 to 1.0 for normalized vectors)
        interpretation: Human-readable interpretation of the score
    """
    chunk: Chunk
    similarity: float
    interpretation: str


@dataclass
class ComparisonResult:
    """
    Complete result of comparing a query against a document.

    This is the primary return type from compare_query_to_document().
    It contains all information needed for downstream analysis or display.

    Attributes:
        query: The original query/concept text
        document_char_count: Total characters in the original document
        document_token_count: Total tokens in the original document
        chunk_count: Number of chunks the document was split into
        max_similarity: Highest similarity score across all chunks
        max_similarity_chunk_index: Index of the chunk with highest similarity
        avg_similarity: Mean similarity across all chunks
        chunk_similarities: Per-chunk similarity results, in document order
        model_name: Name of the embedding model used
        embedding_dim: Dimensionality of the embeddings
    """
    query: str
    document_char_count: int
    document_token_count: int
    chunk_count: int
    max_similarity: float
    max_similarity_chunk_index: int
    avg_similarity: float
    chunk_similarities: List[ChunkSimilarity]
    model_name: str
    embedding_dim: int

    def get_chunks_above_threshold(self, threshold: float) -> List[ChunkSimilarity]:
        """Return chunks with similarity >= threshold."""
        return [cs for cs in self.chunk_similarities if cs.similarity >= threshold]

    def get_chunks_below_threshold(self, threshold: float) -> List[ChunkSimilarity]:
        """Return chunks with similarity < threshold."""
        return [cs for cs in self.chunk_similarities if cs.similarity < threshold]


# Interpretation thresholds for similarity scores.
# These are heuristics based on empirical observation, not ground truth.
# Users should calibrate based on their specific use case.
SIMILARITY_THRESHOLDS = {
    "strong": 0.80,      # >= 0.80: Strong semantic alignment
    "moderate": 0.65,    # 0.65-0.80: Moderate alignment
    "weak": 0.45,        # 0.45-0.65: Weak/partial alignment
    # < 0.45: Likely off-topic
}


def interpret_similarity(score: float) -> str:
    """
    Convert a similarity score to a human-readable interpretation.

    These interpretations are heuristics, not authoritative labels.
    The thresholds are based on typical sentence-transformer behavior
    but may need calibration for specific use cases.

    Args:
        score: Cosine similarity score (typically 0.0 to 1.0)

    Returns:
        Human-readable interpretation string
    """
    if score >= SIMILARITY_THRESHOLDS["strong"]:
        return "Strong"
    elif score >= SIMILARITY_THRESHOLDS["moderate"]:
        return "Moderate"
    elif score >= SIMILARITY_THRESHOLDS["weak"]:
        return "Weak"
    else:
        return "Off-topic"
