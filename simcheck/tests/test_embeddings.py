"""Tests for embedding generation.

Note: These tests require sentence-transformers to be installed and
will download models on first run (~90MB for default model).
Tests are marked with pytest.mark.slow for optional skipping.
"""

import pytest
import numpy as np
from simcheck.core.embeddings import (
    embed_text,
    embed_texts,
    get_model_info,
    clear_model_cache,
    EmbeddingError,
    DEFAULT_MODEL,
)


# Mark all tests in this module as potentially slow (model loading)
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestEmbedText:
    """Tests for single text embedding."""

    def test_empty_text_raises(self):
        """Empty text should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty"):
            embed_text("")

    def test_whitespace_only_raises(self):
        """Whitespace-only text should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty"):
            embed_text("   \n\t  ")

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        embedding = embed_text("Hello world")
        assert isinstance(embedding, np.ndarray)

    def test_returns_float32(self):
        """Should return float32 dtype."""
        embedding = embed_text("Hello world")
        assert embedding.dtype == np.float32

    def test_returns_correct_dimensions(self):
        """Should return vector with expected dimensions."""
        embedding = embed_text("Hello world")
        model_info = get_model_info()
        assert embedding.shape == (model_info["embedding_dim"],)

    def test_embeddings_are_normalized(self):
        """Embeddings should be L2 normalized (magnitude ~= 1)."""
        embedding = embed_text("Hello world")
        magnitude = np.linalg.norm(embedding)
        assert magnitude == pytest.approx(1.0, abs=0.01)

    def test_deterministic_results(self):
        """Same text should produce same embedding."""
        text = "This is a test sentence."
        emb1 = embed_text(text)
        emb2 = embed_text(text)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        emb1 = embed_text("Hello world")
        emb2 = embed_text("Goodbye universe")
        assert not np.allclose(emb1, emb2)

    def test_similar_texts_similar_embeddings(self):
        """Semantically similar texts should have similar embeddings."""
        emb1 = embed_text("The cat sat on the mat")
        emb2 = embed_text("A cat is sitting on a rug")
        # Cosine similarity (dot product for normalized vectors)
        similarity = np.dot(emb1, emb2)
        # Should be reasonably similar (> 0.5)
        assert similarity > 0.5


class TestEmbedTexts:
    """Tests for batch text embedding."""

    def test_empty_list_raises(self):
        """Empty list should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty"):
            embed_texts([])

    def test_list_with_empty_text_raises(self):
        """List containing empty text should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="index 1"):
            embed_texts(["valid text", "", "also valid"])

    def test_returns_list_of_arrays(self):
        """Should return list of numpy arrays."""
        embeddings = embed_texts(["Hello", "World"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(e, np.ndarray) for e in embeddings)

    def test_preserves_order(self):
        """Embeddings should be in same order as input texts."""
        # Use very distinct texts so order is clear
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning and artificial intelligence",
            "Cooking recipes for Italian pasta dishes",
        ]
        embeddings = embed_texts(texts)

        # Verify order: each batch embedding should be most similar to its own text
        # when compared against all individual embeddings
        individual_embs = [embed_text(t) for t in texts]

        for i in range(len(texts)):
            # Find which individual embedding is most similar to batch embedding i
            similarities = [np.dot(embeddings[i], ind) for ind in individual_embs]
            best_match = np.argmax(similarities)
            assert best_match == i, f"Batch embedding {i} best matches individual {best_match}"

    def test_batch_produces_valid_embeddings(self):
        """Batch embedding should produce valid, normalized embeddings."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        batch_embs = embed_texts(texts)

        # Each embedding should be normalized (L2 norm ≈ 1)
        for emb in batch_embs:
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01, f"Embedding not normalized: {norm}"

    def test_single_item_list(self):
        """Single-item list should work."""
        embeddings = embed_texts(["Just one"])
        assert len(embeddings) == 1


class TestGetModelInfo:
    """Tests for model info retrieval."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        info = get_model_info()
        assert isinstance(info, dict)

    def test_contains_model_name(self):
        """Should contain model_name key."""
        info = get_model_info()
        assert "model_name" in info
        assert info["model_name"] == DEFAULT_MODEL

    def test_contains_embedding_dim(self):
        """Should contain embedding_dim key."""
        info = get_model_info()
        assert "embedding_dim" in info
        assert isinstance(info["embedding_dim"], int)
        assert info["embedding_dim"] > 0

    def test_default_model_dimensions(self):
        """Default model (bge-base-en-v1.5) should have 768 dimensions."""
        info = get_model_info(DEFAULT_MODEL)
        assert info["embedding_dim"] == 768


class TestModelCache:
    """Tests for model caching behavior."""

    def test_clear_model_cache(self):
        """clear_model_cache should not raise."""
        # Load a model first
        embed_text("test")
        # Clear should work without error
        clear_model_cache()

    def test_model_reloads_after_clear(self):
        """Model should reload after cache clear."""
        emb1 = embed_text("test")
        clear_model_cache()
        emb2 = embed_text("test")
        # Should still produce same results
        np.testing.assert_array_almost_equal(emb1, emb2)


class TestDefaultModel:
    """Tests for default model constant."""

    def test_default_model_is_bge(self):
        """DEFAULT_MODEL should be BAAI/bge-base-en-v1.5."""
        assert DEFAULT_MODEL == "BAAI/bge-base-en-v1.5"
