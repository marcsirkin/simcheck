"""
Embedding generation using sentence-transformers.

This module handles converting text into vector embeddings using
local sentence-transformer models. All processing is done locally
with no external API calls.

Model Selection:
The default model (BAAI/bge-base-en-v1.5) is chosen for:
- State-of-the-art semantic similarity quality
- Strong performance on MTEB benchmarks
- 768-dimensional embeddings
- Good balance of accuracy and speed

For faster but lower quality, consider:
- all-MiniLM-L6-v2 (384 dims, faster but less accurate)

For even higher quality:
- BAAI/bge-large-en-v1.5 (1024 dims, slower but more accurate)

Embedding Behavior:
- Embeddings are L2-normalized by sentence-transformers
- This means cosine similarity = dot product
- Values typically range from -1 to 1, but normalized vectors
  comparing similar content usually fall in 0 to 1 range
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from simcheck.core.models import Vector


# Default embedding model - production quality, good accuracy
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# Module-level model cache to avoid reloading
_model_cache: dict[str, SentenceTransformer] = {}


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


def _get_model(model_name: str) -> SentenceTransformer:
    """
    Get or load a sentence-transformer model.

    Models are cached at module level to avoid expensive reloading.
    First load downloads the model if not cached locally by
    sentence-transformers (~/.cache/torch/sentence_transformers/).

    Args:
        model_name: Name of the sentence-transformer model

    Returns:
        Loaded SentenceTransformer model

    Raises:
        EmbeddingError: If model cannot be loaded
    """
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load model '{model_name}': {e}")
    return _model_cache[model_name]


def get_model_info(model_name: str = DEFAULT_MODEL) -> dict:
    """
    Get information about an embedding model.

    Args:
        model_name: Name of the model

    Returns:
        Dict with model_name and embedding_dim
    """
    model = _get_model(model_name)
    return {
        "model_name": model_name,
        "embedding_dim": model.get_sentence_embedding_dimension(),
    }


def embed_text(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> Vector:
    """
    Generate embedding vector for a single text.

    Args:
        text: Text to embed
        model_name: Name of the sentence-transformer model to use

    Returns:
        Embedding vector as numpy array (float32)

    Raises:
        EmbeddingError: If text is empty or embedding fails
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    model = _get_model(model_name)

    try:
        # sentence-transformers returns numpy array by default
        # convert_to_numpy=True is the default, but explicit for clarity
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embedding.astype(np.float32)
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embedding: {e}")


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> List[Vector]:
    """
    Generate embedding vectors for multiple texts.

    This is more efficient than calling embed_text() in a loop
    because it batches the encoding operation.

    Args:
        texts: List of texts to embed
        model_name: Name of the sentence-transformer model to use
        batch_size: Number of texts to process at once

    Returns:
        List of embedding vectors, same order as input texts

    Raises:
        EmbeddingError: If any text is empty or embedding fails
    """
    if not texts:
        raise EmbeddingError("Cannot embed empty list of texts")

    # Validate all texts before processing
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise EmbeddingError(f"Text at index {i} is empty")

    model = _get_model(model_name)

    try:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,  # No UI in core module
        )
        # Convert to list of individual vectors
        return [emb.astype(np.float32) for emb in embeddings]
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}")


def clear_model_cache() -> None:
    """
    Clear the model cache to free memory.

    Call this if you need to release GPU/CPU memory used by loaded models.
    """
    _model_cache.clear()
