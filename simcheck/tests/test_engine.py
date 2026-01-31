"""Tests for the main comparison engine.

These are integration-style tests that exercise the full pipeline.
They require sentence-transformers and will download models on first run.
"""

import pytest
from simcheck.core.engine import (
    compare_query_to_document,
    compare_query_to_chunks,
    ComparisonError,
)
from simcheck.core.models import ComparisonResult, ChunkSimilarity
from simcheck.core.chunker import chunk_document


class TestCompareQueryToDocument:
    """Tests for the main comparison function."""

    # Sample documents for testing
    BASEBALL_DOC = """
    Major League Baseball is the oldest professional sports league in the United States.
    The MLB season runs from late March through October. Teams compete in the American
    League and National League. The World Series determines the champion each year.
    Baseball has been called America's pastime for over a century.
    """

    UNRELATED_DOC = """
    Quantum computing represents a fundamentally different approach to computation.
    Unlike classical computers that use bits, quantum computers use qubits.
    These qubits can exist in superposition states. This enables certain calculations
    to be performed exponentially faster than on traditional hardware.
    """

    def test_empty_query_raises(self):
        """Empty query should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Qq]uery.*empty"):
            compare_query_to_document("", self.BASEBALL_DOC)

    def test_whitespace_query_raises(self):
        """Whitespace-only query should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Qq]uery.*empty"):
            compare_query_to_document("   \n\t", self.BASEBALL_DOC)

    def test_none_query_raises(self):
        """None query should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Qq]uery.*None"):
            compare_query_to_document(None, self.BASEBALL_DOC)

    def test_empty_document_raises(self):
        """Empty document should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Dd]ocument.*empty"):
            compare_query_to_document("test query", "")

    def test_whitespace_document_raises(self):
        """Whitespace-only document should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Dd]ocument.*empty"):
            compare_query_to_document("test query", "   \n\t")

    def test_none_document_raises(self):
        """None document should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Dd]ocument.*None"):
            compare_query_to_document("test query", None)

    def test_returns_comparison_result(self):
        """Should return a ComparisonResult object."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        assert isinstance(result, ComparisonResult)

    def test_result_contains_query(self):
        """Result should contain the original query."""
        query = "Major League Baseball"
        result = compare_query_to_document(query, self.BASEBALL_DOC)
        assert result.query == query

    def test_result_has_chunk_count(self):
        """Result should have positive chunk count."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        assert result.chunk_count >= 1

    def test_result_has_similarities(self):
        """Result should have chunk_similarities list."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        assert len(result.chunk_similarities) == result.chunk_count

    def test_chunk_similarities_are_chunk_similarity_objects(self):
        """Each item in chunk_similarities should be ChunkSimilarity."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        for cs in result.chunk_similarities:
            assert isinstance(cs, ChunkSimilarity)

    def test_similarities_in_valid_range(self):
        """All similarity scores should be in [-1, 1] range."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        for cs in result.chunk_similarities:
            assert -1.0 <= cs.similarity <= 1.0

    def test_max_similarity_is_max(self):
        """max_similarity should equal the maximum chunk similarity."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        actual_max = max(cs.similarity for cs in result.chunk_similarities)
        assert result.max_similarity == pytest.approx(actual_max)

    def test_max_similarity_chunk_index_is_correct(self):
        """max_similarity_chunk_index should point to highest scoring chunk."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        max_chunk = result.chunk_similarities[result.max_similarity_chunk_index]
        assert max_chunk.similarity == pytest.approx(result.max_similarity)

    def test_avg_similarity_is_average(self):
        """avg_similarity should equal the mean of chunk similarities."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        actual_avg = sum(cs.similarity for cs in result.chunk_similarities) / len(result.chunk_similarities)
        assert result.avg_similarity == pytest.approx(actual_avg)

    def test_related_query_higher_similarity(self):
        """Related query should have higher similarity than unrelated."""
        related = compare_query_to_document("baseball sports league", self.BASEBALL_DOC)
        unrelated = compare_query_to_document("quantum computing qubits", self.BASEBALL_DOC)

        assert related.max_similarity > unrelated.max_similarity
        assert related.avg_similarity > unrelated.avg_similarity

    def test_result_has_model_info(self):
        """Result should include model name and embedding dimensions."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        assert result.model_name is not None
        assert result.embedding_dim > 0

    def test_result_has_document_stats(self):
        """Result should include document statistics."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        assert result.document_char_count > 0
        assert result.document_token_count > 0

    def test_chunks_have_interpretations(self):
        """Each chunk similarity should have an interpretation string."""
        result = compare_query_to_document("baseball", self.BASEBALL_DOC)
        for cs in result.chunk_similarities:
            assert cs.interpretation in ["Strong", "Moderate", "Weak", "Off-topic"]

    def test_short_document_single_chunk(self):
        """Very short document should produce single chunk."""
        short_doc = "Baseball is great."
        result = compare_query_to_document("baseball", short_doc)
        assert result.chunk_count == 1


class TestCompareQueryToChunks:
    """Tests for comparing query to pre-chunked document."""

    SAMPLE_DOC = """
    First paragraph about technology and computers.
    Second paragraph about nature and wildlife.
    Third paragraph about sports and athletics.
    """

    def test_empty_query_raises(self):
        """Empty query should raise ComparisonError."""
        chunks = chunk_document(self.SAMPLE_DOC)
        with pytest.raises(ComparisonError, match="[Qq]uery.*empty"):
            compare_query_to_chunks("", chunks)

    def test_empty_chunks_raises(self):
        """Empty chunks list should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="[Cc]hunks.*empty"):
            compare_query_to_chunks("test", [])

    def test_returns_list_of_chunk_similarities(self):
        """Should return list of ChunkSimilarity objects."""
        chunks = chunk_document(self.SAMPLE_DOC)
        results = compare_query_to_chunks("technology", chunks)
        assert isinstance(results, list)
        assert all(isinstance(r, ChunkSimilarity) for r in results)

    def test_preserves_chunk_order(self):
        """Results should be in same order as input chunks."""
        chunks = chunk_document(self.SAMPLE_DOC)
        results = compare_query_to_chunks("test", chunks)

        for i, result in enumerate(results):
            assert result.chunk.index == chunks[i].index

    def test_same_results_as_full_comparison(self):
        """Should produce same similarities as compare_query_to_document."""
        query = "sports athletics"
        doc = self.SAMPLE_DOC

        full_result = compare_query_to_document(query, doc)
        chunks = chunk_document(doc)
        chunk_results = compare_query_to_chunks(query, chunks)

        for full, chunked in zip(full_result.chunk_similarities, chunk_results):
            assert full.similarity == pytest.approx(chunked.similarity, abs=0.001)


class TestDeterminism:
    """Tests to verify results are deterministic."""

    SAMPLE_DOC = "This is a test document about machine learning and AI."

    def test_same_inputs_same_results(self):
        """Same query and document should produce identical results."""
        query = "artificial intelligence"

        result1 = compare_query_to_document(query, self.SAMPLE_DOC)
        result2 = compare_query_to_document(query, self.SAMPLE_DOC)

        assert result1.max_similarity == result2.max_similarity
        assert result1.avg_similarity == result2.avg_similarity
        assert result1.chunk_count == result2.chunk_count

        for cs1, cs2 in zip(result1.chunk_similarities, result2.chunk_similarities):
            assert cs1.similarity == pytest.approx(cs2.similarity)
