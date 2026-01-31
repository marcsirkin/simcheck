"""Tests for cosine similarity calculations."""

import pytest
import numpy as np
from simcheck.core.similarity import (
    cosine_similarity,
    cosine_similarity_normalized,
    compute_similarities,
)


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_similar_vectors(self):
        """Similar vectors should have high positive similarity."""
        vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec_b = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        sim = cosine_similarity(vec_a, vec_b)
        assert sim > 0.99  # Very similar

    def test_different_magnitudes(self):
        """Vectors with same direction but different magnitudes should be 1.0."""
        vec_a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        vec_b = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self):
        """Vectors with different dimensions should raise ValueError."""
        vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec_b = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity(vec_a, vec_b)

    def test_zero_length_vector_raises(self):
        """Zero-length vectors should raise ValueError."""
        vec_a = np.array([], dtype=np.float32)
        vec_b = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="zero-length"):
            cosine_similarity(vec_a, vec_b)

    def test_zero_magnitude_vector_raises(self):
        """Zero-magnitude vector should raise ValueError."""
        vec_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="zero-magnitude"):
            cosine_similarity(vec_a, vec_b)

    def test_result_clamped_to_valid_range(self):
        """Result should be clamped to [-1, 1] for floating point errors."""
        # This is hard to trigger naturally, but the function should handle it
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = cosine_similarity(vec, vec)
        assert -1.0 <= sim <= 1.0


class TestCosineSimilarityNormalized:
    """Tests for the fast path with pre-normalized vectors."""

    def test_normalized_identical(self):
        """Identical normalized vectors should have similarity 1.0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Already unit length
        assert cosine_similarity_normalized(vec, vec) == pytest.approx(1.0)

    def test_normalized_orthogonal(self):
        """Orthogonal normalized vectors should have similarity 0.0."""
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity_normalized(vec_a, vec_b) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        """Should raise on dimension mismatch."""
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity_normalized(vec_a, vec_b)

    def test_matches_regular_for_normalized(self):
        """Should match regular cosine_similarity for normalized vectors."""
        # Create and normalize vectors
        vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        vec_a_norm = vec_a / np.linalg.norm(vec_a)
        vec_b_norm = vec_b / np.linalg.norm(vec_b)

        sim_regular = cosine_similarity(vec_a_norm, vec_b_norm)
        sim_fast = cosine_similarity_normalized(vec_a_norm, vec_b_norm)

        assert sim_fast == pytest.approx(sim_regular)


class TestComputeSimilarities:
    """Tests for batch similarity computation."""

    def test_empty_document_vecs(self):
        """Empty document list should return empty list."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = compute_similarities(query, [])
        assert result == []

    def test_single_document(self):
        """Single document should return single similarity."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        doc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = compute_similarities(query, [doc])
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_multiple_documents(self):
        """Multiple documents should return list of similarities."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Same
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Orthogonal
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Opposite
        ]
        result = compute_similarities(query, docs)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(-1.0)

    def test_preserves_order(self):
        """Results should be in same order as input documents."""
        query = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        # Documents with varying similarity to query
        docs = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Partial match
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Partial match
            np.array([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2),  # Exact match
        ]

        result = compute_similarities(query, docs)

        # Third doc should have highest similarity
        assert result[2] > result[0]
        assert result[2] > result[1]

    def test_assume_normalized_parameter(self):
        """assume_normalized=False should use full cosine calculation."""
        query = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # Not normalized
        doc = np.array([3.0, 0.0, 0.0], dtype=np.float32)  # Not normalized

        # Both should give same result since they point same direction
        result_normalized = compute_similarities(query, [doc], assume_normalized=True)
        result_full = compute_similarities(query, [doc], assume_normalized=False)

        # The full calculation should still give 1.0 (same direction)
        assert result_full[0] == pytest.approx(1.0)
