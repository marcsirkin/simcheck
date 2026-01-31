"""Tests for data models and interpretation functions."""

import pytest
from simcheck.core.models import (
    Chunk,
    ChunkSimilarity,
    ComparisonResult,
    interpret_similarity,
    SIMILARITY_THRESHOLDS,
)


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should store all provided attributes."""
        chunk = Chunk(
            index=0,
            text="Hello world",
            char_start=0,
            char_end=11,
            token_count=2,
        )
        assert chunk.index == 0
        assert chunk.text == "Hello world"
        assert chunk.char_start == 0
        assert chunk.char_end == 11
        assert chunk.token_count == 2

    def test_chunk_char_count_property(self):
        """char_count should return length of text."""
        chunk = Chunk(
            index=0,
            text="Hello world",
            char_start=0,
            char_end=11,
            token_count=2,
        )
        assert chunk.char_count == 11

    def test_chunk_empty_text(self):
        """Chunk with empty text should have char_count of 0."""
        chunk = Chunk(
            index=0,
            text="",
            char_start=0,
            char_end=0,
            token_count=0,
        )
        assert chunk.char_count == 0


class TestChunkSimilarity:
    """Tests for the ChunkSimilarity dataclass."""

    def test_chunk_similarity_creation(self):
        """ChunkSimilarity should store chunk, similarity, and interpretation."""
        chunk = Chunk(index=0, text="test", char_start=0, char_end=4, token_count=1)
        cs = ChunkSimilarity(
            chunk=chunk,
            similarity=0.85,
            interpretation="Strong",
        )
        assert cs.chunk == chunk
        assert cs.similarity == 0.85
        assert cs.interpretation == "Strong"


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def _make_result(self) -> ComparisonResult:
        """Helper to create a test ComparisonResult."""
        chunks = [
            Chunk(index=0, text="chunk 0", char_start=0, char_end=7, token_count=2),
            Chunk(index=1, text="chunk 1", char_start=8, char_end=15, token_count=2),
            Chunk(index=2, text="chunk 2", char_start=16, char_end=23, token_count=2),
        ]
        chunk_sims = [
            ChunkSimilarity(chunk=chunks[0], similarity=0.9, interpretation="Strong"),
            ChunkSimilarity(chunk=chunks[1], similarity=0.5, interpretation="Weak"),
            ChunkSimilarity(chunk=chunks[2], similarity=0.3, interpretation="Off-topic"),
        ]
        return ComparisonResult(
            query="test query",
            document_char_count=23,
            document_token_count=6,
            chunk_count=3,
            max_similarity=0.9,
            max_similarity_chunk_index=0,
            avg_similarity=0.567,
            chunk_similarities=chunk_sims,
            model_name="test-model",
            embedding_dim=384,
        )

    def test_comparison_result_creation(self):
        """ComparisonResult should store all attributes."""
        result = self._make_result()
        assert result.query == "test query"
        assert result.chunk_count == 3
        assert result.max_similarity == 0.9
        assert result.avg_similarity == 0.567
        assert len(result.chunk_similarities) == 3

    def test_get_chunks_above_threshold(self):
        """Should return chunks with similarity >= threshold."""
        result = self._make_result()
        above = result.get_chunks_above_threshold(0.8)
        assert len(above) == 1
        assert above[0].similarity == 0.9

    def test_get_chunks_above_threshold_inclusive(self):
        """Threshold should be inclusive (>=)."""
        result = self._make_result()
        above = result.get_chunks_above_threshold(0.5)
        assert len(above) == 2  # 0.9 and 0.5

    def test_get_chunks_below_threshold(self):
        """Should return chunks with similarity < threshold."""
        result = self._make_result()
        below = result.get_chunks_below_threshold(0.5)
        assert len(below) == 1
        assert below[0].similarity == 0.3

    def test_get_chunks_below_threshold_exclusive(self):
        """Threshold should be exclusive (<)."""
        result = self._make_result()
        below = result.get_chunks_below_threshold(0.9)
        assert len(below) == 2  # 0.5 and 0.3


class TestInterpretSimilarity:
    """Tests for the interpret_similarity function."""

    def test_strong_threshold(self):
        """Scores >= 0.80 should be 'Strong'."""
        assert interpret_similarity(0.80) == "Strong"
        assert interpret_similarity(0.85) == "Strong"
        assert interpret_similarity(0.99) == "Strong"
        assert interpret_similarity(1.0) == "Strong"

    def test_moderate_threshold(self):
        """Scores in [0.65, 0.80) should be 'Moderate'."""
        assert interpret_similarity(0.65) == "Moderate"
        assert interpret_similarity(0.70) == "Moderate"
        assert interpret_similarity(0.79) == "Moderate"

    def test_weak_threshold(self):
        """Scores in [0.45, 0.65) should be 'Weak'."""
        assert interpret_similarity(0.45) == "Weak"
        assert interpret_similarity(0.50) == "Weak"
        assert interpret_similarity(0.64) == "Weak"

    def test_off_topic_threshold(self):
        """Scores < 0.45 should be 'Off-topic'."""
        assert interpret_similarity(0.44) == "Off-topic"
        assert interpret_similarity(0.30) == "Off-topic"
        assert interpret_similarity(0.0) == "Off-topic"

    def test_negative_scores(self):
        """Negative scores should be 'Off-topic'."""
        assert interpret_similarity(-0.1) == "Off-topic"
        assert interpret_similarity(-1.0) == "Off-topic"

    def test_thresholds_are_configurable(self):
        """SIMILARITY_THRESHOLDS dict should be accessible."""
        assert "strong" in SIMILARITY_THRESHOLDS
        assert "moderate" in SIMILARITY_THRESHOLDS
        assert "weak" in SIMILARITY_THRESHOLDS
        assert SIMILARITY_THRESHOLDS["strong"] == 0.80
