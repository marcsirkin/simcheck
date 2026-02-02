"""
Main comparison engine orchestrating the full analysis pipeline.

This module provides the high-level API for comparing a query/concept
against a document. It coordinates:
1. Document chunking
2. Embedding generation
3. Similarity computation
4. Result aggregation

The primary entry point is compare_query_to_document(), which takes
raw text inputs and returns a structured ComparisonResult.

Design Principles:
- Single responsibility: orchestration only, delegates to specialized modules
- Fail-fast: validate inputs early, surface clear errors
- Deterministic: same inputs produce same outputs
- No side effects: pure computation, no persistence or external calls
"""

from typing import List, Optional, Union

from simcheck.core.models import (
    Chunk,
    ChunkSimilarity,
    ComparisonResult,
    Vector,
    interpret_similarity,
)
from simcheck.core.chunker import (
    chunk_document,
    get_total_tokens,
    ChunkingError,
    ChunkingStrategy,
    ChunkingConfig,
)
from simcheck.core.embeddings import (
    embed_text,
    embed_texts,
    get_model_info,
    EmbeddingError,
    DEFAULT_MODEL,
)
from simcheck.core.similarity import compute_similarities


class ComparisonError(Exception):
    """Raised when the comparison pipeline fails."""
    pass


def _validate_inputs(query: str, document: str) -> None:
    """
    Validate query and document inputs.

    Args:
        query: The query/concept text
        document: The document text

    Raises:
        ComparisonError: If inputs are invalid
    """
    if query is None:
        raise ComparisonError("Query cannot be None")

    if document is None:
        raise ComparisonError("Document cannot be None")

    if not query.strip():
        raise ComparisonError("Query is empty or contains only whitespace")

    if not document.strip():
        raise ComparisonError("Document is empty or contains only whitespace")


def compare_query_to_document(
    query: str,
    document: str,
    model_name: str = DEFAULT_MODEL,
    chunking_strategy: Union[ChunkingStrategy, str] = ChunkingStrategy.FLAT,
    chunking_config: Optional[ChunkingConfig] = None,
) -> ComparisonResult:
    """
    Compare a query/concept against a document and return semantic similarity analysis.

    This is the main entry point for the comparison engine. It:
    1. Validates inputs
    2. Chunks the document into semantic segments
    3. Generates embeddings for query and all chunks
    4. Computes cosine similarity between query and each chunk
    5. Aggregates results into a structured response

    Args:
        query: Short concept or phrase to search for (e.g., "Major League Baseball")
        document: Longer text to analyze (e.g., blog post, article)
        model_name: sentence-transformer model to use for embeddings
        chunking_strategy: Strategy for chunking (FLAT, MARKDOWN, HTML, AUTO)
                          Can be ChunkingStrategy enum or string value
        chunking_config: Optional configuration for hierarchical chunking

    Returns:
        ComparisonResult with max/avg similarity, per-chunk scores, and metadata

    Raises:
        ComparisonError: If inputs are invalid or processing fails

    Example:
        >>> # Default flat chunking
        >>> result = compare_query_to_document(
        ...     query="artificial intelligence",
        ...     document="AI is transforming industries. Machine learning models..."
        ... )
        >>> print(f"Max similarity: {result.max_similarity:.2f}")
        >>>
        >>> # Hierarchical chunking for Markdown documents
        >>> result = compare_query_to_document(
        ...     query="machine learning",
        ...     document=markdown_article,
        ...     chunking_strategy=ChunkingStrategy.MARKDOWN,
        ... )
    """
    # Step 0: Validate inputs
    _validate_inputs(query, document)

    # Normalize strategy to enum
    if isinstance(chunking_strategy, str):
        try:
            chunking_strategy = ChunkingStrategy(chunking_strategy)
        except ValueError:
            raise ComparisonError(f"Invalid chunking strategy: {chunking_strategy}")

    # Step 1: Chunk the document
    try:
        chunks: List[Chunk] = chunk_document(
            document,
            strategy=chunking_strategy,
            config=chunking_config,
        )
    except ChunkingError as e:
        raise ComparisonError(f"Failed to chunk document: {e}")

    # Step 2: Get model info for metadata
    model_info = get_model_info(model_name)

    # Step 3: Generate embeddings
    try:
        # Embed query
        query_embedding: Vector = embed_text(query, model_name=model_name)

        # Embed all chunks at once (more efficient than one-by-one)
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings: List[Vector] = embed_texts(chunk_texts, model_name=model_name)
    except EmbeddingError as e:
        raise ComparisonError(f"Failed to generate embeddings: {e}")

    # Step 4: Compute similarities
    similarities: List[float] = compute_similarities(
        query_embedding,
        chunk_embeddings,
        assume_normalized=True,  # sentence-transformers normalizes by default
    )

    # Step 5: Build per-chunk results with interpretations
    chunk_similarities: List[ChunkSimilarity] = []
    for chunk, similarity in zip(chunks, similarities):
        chunk_similarities.append(ChunkSimilarity(
            chunk=chunk,
            similarity=similarity,
            interpretation=interpret_similarity(similarity),
        ))

    # Step 6: Compute aggregate metrics
    max_similarity = max(similarities)
    max_similarity_index = similarities.index(max_similarity)
    avg_similarity = sum(similarities) / len(similarities)
    total_tokens = get_total_tokens(chunks)

    # Step 7: Build and return result
    return ComparisonResult(
        query=query,
        document_char_count=len(document),
        document_token_count=total_tokens,
        chunk_count=len(chunks),
        max_similarity=max_similarity,
        max_similarity_chunk_index=max_similarity_index,
        avg_similarity=avg_similarity,
        chunk_similarities=chunk_similarities,
        model_name=model_info["model_name"],
        embedding_dim=model_info["embedding_dim"],
    )


def compare_query_to_chunks(
    query: str,
    chunks: List[Chunk],
    model_name: str = DEFAULT_MODEL,
) -> List[ChunkSimilarity]:
    """
    Compare a query against pre-chunked document segments.

    Use this when you want to re-run queries against the same document
    without re-chunking. Note that embeddings are regenerated each time;
    for caching embeddings, manage that at a higher level.

    Args:
        query: Short concept or phrase to search for
        chunks: Pre-computed chunks from chunk_document()
        model_name: sentence-transformer model to use

    Returns:
        List of ChunkSimilarity objects in chunk order

    Raises:
        ComparisonError: If query is invalid or embedding fails
    """
    if not query or not query.strip():
        raise ComparisonError("Query is empty or contains only whitespace")

    if not chunks:
        raise ComparisonError("Chunks list is empty")

    try:
        query_embedding = embed_text(query, model_name=model_name)
        chunk_embeddings = embed_texts(
            [c.text for c in chunks],
            model_name=model_name,
        )
    except EmbeddingError as e:
        raise ComparisonError(f"Failed to generate embeddings: {e}")

    similarities = compute_similarities(query_embedding, chunk_embeddings)

    return [
        ChunkSimilarity(
            chunk=chunk,
            similarity=sim,
            interpretation=interpret_similarity(sim),
        )
        for chunk, sim in zip(chunks, similarities)
    ]
