"""
Cosine similarity calculations.

This module provides the core similarity computation between embedding vectors.
We use only cosine similarity as the metric (per requirements).

Mathematical Background:
Cosine similarity measures the angle between two vectors:
    cos(theta) = (A . B) / (||A|| * ||B||)

For L2-normalized vectors (which sentence-transformers produces by default),
this simplifies to just the dot product:
    cos(theta) = A . B

Interpretation:
- 1.0: Identical direction (same semantic meaning)
- 0.0: Orthogonal (unrelated)
- -1.0: Opposite direction (rare for text embeddings)

In practice, text embeddings from sentence-transformers typically
produce similarities in the 0.0 to 1.0 range for most content,
with negative values being uncommon.
"""

from typing import List
import numpy as np

from simcheck.core.models import Vector


def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """
    Compute cosine similarity between two vectors.

    For normalized vectors (default from sentence-transformers),
    this is equivalent to the dot product.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Cosine similarity score (typically -1.0 to 1.0)

    Raises:
        ValueError: If vectors have different dimensions or are zero-length
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {vec_a.shape} vs {vec_b.shape}"
        )

    if vec_a.size == 0:
        raise ValueError("Cannot compute similarity of zero-length vectors")

    # Compute norms
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Handle zero vectors (shouldn't happen with real embeddings, but be safe)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute similarity with zero-magnitude vector")

    # Cosine similarity formula
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)

    # Clamp to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_similarity_normalized(vec_a: Vector, vec_b: Vector) -> float:
    """
    Compute cosine similarity for pre-normalized vectors (fast path).

    Use this when you know vectors are already L2-normalized,
    as it skips the normalization step. sentence-transformers
    produces normalized vectors by default.

    Args:
        vec_a: First normalized embedding vector
        vec_b: Second normalized embedding vector

    Returns:
        Cosine similarity score (typically -1.0 to 1.0)
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {vec_a.shape} vs {vec_b.shape}"
        )

    # For normalized vectors, cosine similarity = dot product
    similarity = np.dot(vec_a, vec_b)

    # Clamp to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def compute_similarities(
    query_vec: Vector,
    document_vecs: List[Vector],
    assume_normalized: bool = True,
) -> List[float]:
    """
    Compute cosine similarity between a query vector and multiple document vectors.

    Args:
        query_vec: Query embedding vector
        document_vecs: List of document chunk embedding vectors
        assume_normalized: If True, use fast dot-product path

    Returns:
        List of similarity scores, same order as document_vecs
    """
    if not document_vecs:
        return []

    sim_func = cosine_similarity_normalized if assume_normalized else cosine_similarity

    return [sim_func(query_vec, doc_vec) for doc_vec in document_vecs]
