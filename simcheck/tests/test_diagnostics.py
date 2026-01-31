"""Tests for chunk-level semantic diagnostics (Feature 2)."""

import pytest
from simcheck.core.models import (
    Chunk,
    ChunkSimilarity,
    ComparisonResult,
    SIMILARITY_THRESHOLDS,
)
from simcheck.core.diagnostics import (
    create_diagnostic_report,
    ChunkDiagnostic,
    DiagnosticReport,
    DiagnosticSummary,
    SortOrder,
    ConceptCoverageScore,
    compute_concept_coverage_score,
    interpret_coverage_score,
    COVERAGE_WEIGHTS,
    COVERAGE_INTERPRETATION,
)


# -----------------------------------------------------------------------------
# Test fixtures - create mock ComparisonResult objects
# -----------------------------------------------------------------------------

def make_chunk(index: int, text: str, token_count: int = 10) -> Chunk:
    """Helper to create a Chunk."""
    return Chunk(
        index=index,
        text=text,
        char_start=index * 100,
        char_end=(index * 100) + len(text),
        token_count=token_count,
    )


def make_chunk_similarity(
    index: int,
    text: str,
    similarity: float,
    interpretation: str,
) -> ChunkSimilarity:
    """Helper to create a ChunkSimilarity."""
    chunk = make_chunk(index, text)
    return ChunkSimilarity(
        chunk=chunk,
        similarity=similarity,
        interpretation=interpretation,
    )


def make_comparison_result(
    chunk_similarities: list[ChunkSimilarity],
    query: str = "test query",
) -> ComparisonResult:
    """Helper to create a ComparisonResult from chunk similarities."""
    scores = [cs.similarity for cs in chunk_similarities]
    max_sim = max(scores)
    max_idx = scores.index(max_sim)
    avg_sim = sum(scores) / len(scores)
    total_chars = sum(cs.chunk.char_count for cs in chunk_similarities)
    total_tokens = sum(cs.chunk.token_count for cs in chunk_similarities)

    return ComparisonResult(
        query=query,
        document_char_count=total_chars,
        document_token_count=total_tokens,
        chunk_count=len(chunk_similarities),
        max_similarity=max_sim,
        max_similarity_chunk_index=max_idx,
        avg_similarity=avg_sim,
        chunk_similarities=chunk_similarities,
        model_name="test-model",
        embedding_dim=384,
    )


@pytest.fixture
def sample_result() -> ComparisonResult:
    """Create a sample ComparisonResult with varied similarity scores."""
    chunk_sims = [
        make_chunk_similarity(0, "First chunk about baseball.", 0.85, "Strong"),
        make_chunk_similarity(1, "Second chunk also about sports.", 0.72, "Moderate"),
        make_chunk_similarity(2, "Third chunk drifts to other topics.", 0.48, "Weak"),
        make_chunk_similarity(3, "Fourth chunk is completely off-topic.", 0.25, "Off-topic"),
    ]
    return make_comparison_result(chunk_sims, query="baseball")


@pytest.fixture
def uniform_result() -> ComparisonResult:
    """Create a ComparisonResult where all chunks have same similarity."""
    chunk_sims = [
        make_chunk_similarity(0, "Chunk A", 0.70, "Moderate"),
        make_chunk_similarity(1, "Chunk B", 0.70, "Moderate"),
        make_chunk_similarity(2, "Chunk C", 0.70, "Moderate"),
    ]
    return make_comparison_result(chunk_sims)


@pytest.fixture
def single_chunk_result() -> ComparisonResult:
    """Create a ComparisonResult with just one chunk."""
    chunk_sims = [
        make_chunk_similarity(0, "Only chunk.", 0.60, "Weak"),
    ]
    return make_comparison_result(chunk_sims)


# -----------------------------------------------------------------------------
# Tests for create_diagnostic_report
# -----------------------------------------------------------------------------

class TestCreateDiagnosticReport:
    """Tests for the main factory function."""

    def test_returns_diagnostic_report(self, sample_result):
        """Should return a DiagnosticReport object."""
        report = create_diagnostic_report(sample_result)
        assert isinstance(report, DiagnosticReport)

    def test_preserves_query(self, sample_result):
        """Report should contain original query."""
        report = create_diagnostic_report(sample_result)
        assert report.query == "baseball"

    def test_preserves_model_name(self, sample_result):
        """Report should contain model name."""
        report = create_diagnostic_report(sample_result)
        assert report.model_name == "test-model"

    def test_creates_chunk_diagnostics(self, sample_result):
        """Report should have same number of chunk diagnostics as input."""
        report = create_diagnostic_report(sample_result)
        assert len(report.chunks) == 4

    def test_chunk_diagnostics_are_correct_type(self, sample_result):
        """Each chunk should be a ChunkDiagnostic."""
        report = create_diagnostic_report(sample_result)
        for chunk in report.chunks:
            assert isinstance(chunk, ChunkDiagnostic)

    def test_creates_summary(self, sample_result):
        """Report should have a summary."""
        report = create_diagnostic_report(sample_result)
        assert isinstance(report.summary, DiagnosticSummary)

    def test_empty_result_raises(self):
        """Should raise on empty chunk similarities."""
        empty_result = ComparisonResult(
            query="test",
            document_char_count=0,
            document_token_count=0,
            chunk_count=0,
            max_similarity=0,
            max_similarity_chunk_index=0,
            avg_similarity=0,
            chunk_similarities=[],
            model_name="test",
            embedding_dim=384,
        )
        with pytest.raises(ValueError, match="no chunk"):
            create_diagnostic_report(empty_result)


# -----------------------------------------------------------------------------
# Tests for ChunkDiagnostic
# -----------------------------------------------------------------------------

class TestChunkDiagnostic:
    """Tests for individual chunk diagnostic data."""

    def test_preserves_chunk_index(self, sample_result):
        """Diagnostic should preserve original chunk index."""
        report = create_diagnostic_report(sample_result)
        for i, chunk in enumerate(report.chunks):
            assert chunk.chunk_index == i

    def test_preserves_text(self, sample_result):
        """Diagnostic should preserve chunk text."""
        report = create_diagnostic_report(sample_result)
        assert "baseball" in report.chunks[0].text

    def test_text_preview_truncates(self, sample_result):
        """Text preview should truncate long text."""
        long_text = "x" * 200
        chunk_sims = [make_chunk_similarity(0, long_text, 0.5, "Weak")]
        result = make_comparison_result(chunk_sims)
        report = create_diagnostic_report(result, preview_length=50)

        assert len(report.chunks[0].text_preview) < len(long_text)
        assert report.chunks[0].text_preview.endswith("...")

    def test_text_preview_no_truncate_short(self, sample_result):
        """Short text should not be truncated."""
        report = create_diagnostic_report(sample_result, preview_length=200)
        # First chunk is short, should not have ...
        assert not report.chunks[0].text_preview.endswith("...")

    def test_similarity_preserved(self, sample_result):
        """Diagnostic should preserve similarity score."""
        report = create_diagnostic_report(sample_result)
        assert report.chunks[0].similarity == 0.85
        assert report.chunks[3].similarity == 0.25

    def test_interpretation_preserved(self, sample_result):
        """Diagnostic should preserve interpretation."""
        report = create_diagnostic_report(sample_result)
        assert report.chunks[0].interpretation == "Strong"
        assert report.chunks[3].interpretation == "Off-topic"

    def test_similarity_rank_computed(self, sample_result):
        """Chunks should have correct similarity ranks."""
        report = create_diagnostic_report(sample_result)
        # Highest similarity (0.85) should be rank 1
        assert report.chunks[0].similarity_rank == 1
        # Lowest similarity (0.25) should be rank 4
        assert report.chunks[3].similarity_rank == 4

    def test_normalized_score_in_range(self, sample_result):
        """Normalized scores should be in [0, 1] range."""
        report = create_diagnostic_report(sample_result)
        for chunk in report.chunks:
            assert 0.0 <= chunk.normalized_score <= 1.0

    def test_normalized_score_max_is_one(self, sample_result):
        """Highest scoring chunk should have normalized_score = 1.0."""
        report = create_diagnostic_report(sample_result)
        max_chunk = report.get_max_chunk()
        assert max_chunk.normalized_score == 1.0

    def test_normalized_score_min_is_zero(self, sample_result):
        """Lowest scoring chunk should have normalized_score = 0.0."""
        report = create_diagnostic_report(sample_result)
        min_chunk = report.get_min_chunk()
        assert min_chunk.normalized_score == 0.0

    def test_normalized_score_uniform(self, uniform_result):
        """Uniform scores should all normalize to 0.5."""
        report = create_diagnostic_report(uniform_result)
        for chunk in report.chunks:
            assert chunk.normalized_score == 0.5

    def test_position_percent(self, sample_result):
        """Position percent should reflect document position."""
        report = create_diagnostic_report(sample_result)
        assert report.chunks[0].position_percent == 0.0  # First
        assert report.chunks[3].position_percent == 1.0  # Last

    def test_is_max_flag(self, sample_result):
        """Only highest scoring chunk should have is_max=True."""
        report = create_diagnostic_report(sample_result)
        max_count = sum(1 for c in report.chunks if c.is_max)
        assert max_count == 1
        assert report.chunks[0].is_max  # 0.85 is highest

    def test_is_min_flag(self, sample_result):
        """Only lowest scoring chunk should have is_min=True."""
        report = create_diagnostic_report(sample_result)
        min_count = sum(1 for c in report.chunks if c.is_min)
        assert min_count == 1
        assert report.chunks[3].is_min  # 0.25 is lowest

    def test_threshold_flags(self, sample_result):
        """Threshold flags should reflect similarity thresholds."""
        report = create_diagnostic_report(sample_result)

        # 0.85 - above strong
        assert report.chunks[0].above_strong_threshold
        assert report.chunks[0].above_moderate_threshold
        assert not report.chunks[0].below_weak_threshold

        # 0.25 - below weak
        assert not report.chunks[3].above_strong_threshold
        assert not report.chunks[3].above_moderate_threshold
        assert report.chunks[3].below_weak_threshold


# -----------------------------------------------------------------------------
# Tests for DiagnosticReport sorting
# -----------------------------------------------------------------------------

class TestDiagnosticReportSorting:
    """Tests for sorting methods."""

    def test_by_document_order(self, sample_result):
        """by_document_order should return chunks in index order."""
        report = create_diagnostic_report(sample_result)
        ordered = report.by_document_order()
        indices = [c.chunk_index for c in ordered]
        assert indices == [0, 1, 2, 3]

    def test_by_similarity_descending(self, sample_result):
        """by_similarity_descending should return highest first."""
        report = create_diagnostic_report(sample_result)
        sorted_chunks = report.by_similarity_descending()
        scores = [c.similarity for c in sorted_chunks]
        assert scores == sorted(scores, reverse=True)
        assert sorted_chunks[0].similarity == 0.85

    def test_by_similarity_ascending(self, sample_result):
        """by_similarity_ascending should return lowest first."""
        report = create_diagnostic_report(sample_result)
        sorted_chunks = report.by_similarity_ascending()
        scores = [c.similarity for c in sorted_chunks]
        assert scores == sorted(scores)
        assert sorted_chunks[0].similarity == 0.25

    def test_sorted_by_enum(self, sample_result):
        """sorted_by should work with SortOrder enum."""
        report = create_diagnostic_report(sample_result)

        doc_order = report.sorted_by(SortOrder.DOCUMENT_ORDER)
        assert [c.chunk_index for c in doc_order] == [0, 1, 2, 3]

        desc = report.sorted_by(SortOrder.SIMILARITY_DESC)
        assert desc[0].similarity >= desc[-1].similarity

        asc = report.sorted_by(SortOrder.SIMILARITY_ASC)
        assert asc[0].similarity <= asc[-1].similarity

    def test_sorting_does_not_mutate(self, sample_result):
        """Sorting should return new list, not mutate original."""
        report = create_diagnostic_report(sample_result)
        original_order = [c.chunk_index for c in report.chunks]

        report.by_similarity_descending()
        after_sort = [c.chunk_index for c in report.chunks]

        assert original_order == after_sort


# -----------------------------------------------------------------------------
# Tests for DiagnosticReport filtering
# -----------------------------------------------------------------------------

class TestDiagnosticReportFiltering:
    """Tests for filtering methods."""

    def test_above_threshold(self, sample_result):
        """above_threshold should filter correctly."""
        report = create_diagnostic_report(sample_result)
        above = report.above_threshold(0.70)
        assert len(above) == 2  # 0.85 and 0.72
        assert all(c.similarity >= 0.70 for c in above)

    def test_below_threshold(self, sample_result):
        """below_threshold should filter correctly."""
        report = create_diagnostic_report(sample_result)
        below = report.below_threshold(0.50)
        assert len(below) == 2  # 0.48 and 0.25
        assert all(c.similarity < 0.50 for c in below)

    def test_in_range(self, sample_result):
        """in_range should filter to specified range."""
        report = create_diagnostic_report(sample_result)
        in_range = report.in_range(0.40, 0.75)
        assert len(in_range) == 2  # 0.72 and 0.48
        assert all(0.40 <= c.similarity <= 0.75 for c in in_range)

    def test_strong_chunks(self, sample_result):
        """strong_chunks should return >= 0.80."""
        report = create_diagnostic_report(sample_result)
        strong = report.strong_chunks()
        assert len(strong) == 1
        assert strong[0].similarity == 0.85

    def test_weak_or_off_topic_chunks(self, sample_result):
        """weak_or_off_topic_chunks should return < 0.65."""
        report = create_diagnostic_report(sample_result)
        weak = report.weak_or_off_topic_chunks()
        assert len(weak) == 2  # 0.48 and 0.25

    def test_off_topic_chunks(self, sample_result):
        """off_topic_chunks should return < 0.45."""
        report = create_diagnostic_report(sample_result)
        off_topic = report.off_topic_chunks()
        assert len(off_topic) == 1
        assert off_topic[0].similarity == 0.25

    def test_filter_by_custom_predicate(self, sample_result):
        """filter_by should work with custom predicate."""
        report = create_diagnostic_report(sample_result)
        # Custom: chunks with "about" in text
        filtered = report.filter_by(lambda c: "about" in c.text)
        assert len(filtered) == 2  # Chunks 0 and 1

    def test_filtering_does_not_mutate(self, sample_result):
        """Filtering should return new list, not mutate original."""
        report = create_diagnostic_report(sample_result)
        original_count = len(report.chunks)

        report.above_threshold(0.90)
        after_filter = len(report.chunks)

        assert original_count == after_filter


# -----------------------------------------------------------------------------
# Tests for DiagnosticReport accessors
# -----------------------------------------------------------------------------

class TestDiagnosticReportAccessors:
    """Tests for convenience accessor methods."""

    def test_get_max_chunk(self, sample_result):
        """get_max_chunk should return highest scoring chunk."""
        report = create_diagnostic_report(sample_result)
        max_chunk = report.get_max_chunk()
        assert max_chunk is not None
        assert max_chunk.similarity == 0.85

    def test_get_min_chunk(self, sample_result):
        """get_min_chunk should return lowest scoring chunk."""
        report = create_diagnostic_report(sample_result)
        min_chunk = report.get_min_chunk()
        assert min_chunk is not None
        assert min_chunk.similarity == 0.25

    def test_get_chunk_by_index(self, sample_result):
        """get_chunk should return chunk by index."""
        report = create_diagnostic_report(sample_result)
        chunk = report.get_chunk(2)
        assert chunk is not None
        assert chunk.chunk_index == 2

    def test_get_chunk_invalid_index(self, sample_result):
        """get_chunk should return None for invalid index."""
        report = create_diagnostic_report(sample_result)
        assert report.get_chunk(999) is None

    def test_top_n(self, sample_result):
        """top_n should return top N by similarity."""
        report = create_diagnostic_report(sample_result)
        top2 = report.top_n(2)
        assert len(top2) == 2
        assert top2[0].similarity >= top2[1].similarity
        assert top2[0].similarity == 0.85

    def test_bottom_n(self, sample_result):
        """bottom_n should return bottom N by similarity."""
        report = create_diagnostic_report(sample_result)
        bottom2 = report.bottom_n(2)
        assert len(bottom2) == 2
        assert bottom2[0].similarity <= bottom2[1].similarity
        assert bottom2[0].similarity == 0.25


# -----------------------------------------------------------------------------
# Tests for heatmap data
# -----------------------------------------------------------------------------

class TestHeatmapData:
    """Tests for heatmap-ready data extraction."""

    def test_get_heatmap_data_returns_list(self, sample_result):
        """get_heatmap_data should return list of dicts."""
        report = create_diagnostic_report(sample_result)
        data = report.get_heatmap_data()
        assert isinstance(data, list)
        assert len(data) == 4
        assert all(isinstance(d, dict) for d in data)

    def test_heatmap_data_has_required_keys(self, sample_result):
        """Each heatmap data dict should have required keys."""
        report = create_diagnostic_report(sample_result)
        data = report.get_heatmap_data()
        for d in data:
            assert "index" in d
            assert "score" in d
            assert "raw_score" in d
            assert "interpretation" in d

    def test_heatmap_data_in_document_order(self, sample_result):
        """Heatmap data should be in document order."""
        report = create_diagnostic_report(sample_result)
        data = report.get_heatmap_data()
        indices = [d["index"] for d in data]
        assert indices == [0, 1, 2, 3]

    def test_get_similarity_sequence(self, sample_result):
        """get_similarity_sequence should return scores in order."""
        report = create_diagnostic_report(sample_result)
        seq = report.get_similarity_sequence()
        assert seq == [0.85, 0.72, 0.48, 0.25]

    def test_get_normalized_sequence(self, sample_result):
        """get_normalized_sequence should return normalized scores."""
        report = create_diagnostic_report(sample_result)
        seq = report.get_normalized_sequence()
        assert len(seq) == 4
        assert seq[0] == 1.0  # Max normalized to 1
        assert seq[3] == 0.0  # Min normalized to 0


# -----------------------------------------------------------------------------
# Tests for DiagnosticSummary
# -----------------------------------------------------------------------------

class TestDiagnosticSummary:
    """Tests for aggregate statistics."""

    def test_total_chunks(self, sample_result):
        """Summary should have correct total chunks."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.total_chunks == 4

    def test_max_similarity(self, sample_result):
        """Summary should have correct max similarity."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.max_similarity == 0.85

    def test_min_similarity(self, sample_result):
        """Summary should have correct min similarity."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.min_similarity == 0.25

    def test_avg_similarity(self, sample_result):
        """Summary should have correct average similarity."""
        report = create_diagnostic_report(sample_result)
        expected = (0.85 + 0.72 + 0.48 + 0.25) / 4
        assert report.summary.avg_similarity == pytest.approx(expected)

    def test_median_similarity(self, sample_result):
        """Summary should have correct median similarity."""
        report = create_diagnostic_report(sample_result)
        # Sorted: 0.25, 0.48, 0.72, 0.85 -> median = (0.48 + 0.72) / 2 = 0.60
        assert report.summary.median_similarity == pytest.approx(0.60)

    def test_threshold_counts(self, sample_result):
        """Summary should have correct threshold counts."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.chunks_strong == 1      # 0.85
        assert report.summary.chunks_moderate == 1    # 0.72
        assert report.summary.chunks_weak == 1        # 0.48
        assert report.summary.chunks_off_topic == 1   # 0.25

    def test_percent_strong(self, sample_result):
        """Summary should have correct percent strong."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.percent_strong == pytest.approx(25.0)  # 1/4

    def test_percent_on_topic(self, sample_result):
        """Summary should have correct percent on-topic."""
        report = create_diagnostic_report(sample_result)
        # On-topic = not off-topic = 3/4 = 75%
        assert report.summary.percent_on_topic == pytest.approx(75.0)

    def test_document_stats(self, sample_result):
        """Summary should have document stats."""
        report = create_diagnostic_report(sample_result)
        assert report.summary.document_char_count > 0
        assert report.summary.document_token_count > 0


# -----------------------------------------------------------------------------
# Tests for edge cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_chunk(self, single_chunk_result):
        """Should handle single-chunk documents."""
        report = create_diagnostic_report(single_chunk_result)
        assert len(report.chunks) == 1
        assert report.chunks[0].is_max
        assert report.chunks[0].is_min
        assert report.chunks[0].similarity_rank == 1

    def test_single_chunk_position_percent(self, single_chunk_result):
        """Single chunk should have position_percent = 0."""
        report = create_diagnostic_report(single_chunk_result)
        assert report.chunks[0].position_percent == 0.0

    def test_uniform_scores_ranking(self, uniform_result):
        """Uniform scores should all get different ranks (tiebreaker by index)."""
        report = create_diagnostic_report(uniform_result)
        ranks = {c.similarity_rank for c in report.chunks}
        # All ranks should be unique
        assert len(ranks) == 3

    def test_uniform_scores_max_min(self, uniform_result):
        """Uniform scores: multiple chunks may be max/min."""
        report = create_diagnostic_report(uniform_result)
        max_count = sum(1 for c in report.chunks if c.is_max)
        min_count = sum(1 for c in report.chunks if c.is_min)
        # All have same score, so all are both max and min
        assert max_count == 3
        assert min_count == 3


# -----------------------------------------------------------------------------
# Tests for Concept Coverage Score (Feature 4)
# -----------------------------------------------------------------------------

class TestConceptCoverageScoreComputation:
    """Tests for compute_concept_coverage_score function."""

    def test_all_strong_chunks(self):
        """All strong chunks should yield CCS = 100."""
        ccs = compute_concept_coverage_score(
            chunks_strong=5,
            chunks_moderate=0,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        assert ccs.score == 100.0
        assert ccs.interpretation == "Strong"

    def test_all_off_topic_chunks(self):
        """All off-topic chunks should yield CCS = 0."""
        ccs = compute_concept_coverage_score(
            chunks_strong=0,
            chunks_moderate=0,
            chunks_weak=0,
            chunks_off_topic=5,
        )
        assert ccs.score == 0.0
        assert ccs.interpretation == "Low"

    def test_all_moderate_chunks(self):
        """All moderate chunks should yield CCS = 60."""
        ccs = compute_concept_coverage_score(
            chunks_strong=0,
            chunks_moderate=4,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        # (4 * 0.6) / 4 * 100 = 60
        assert ccs.score == 60.0
        assert ccs.interpretation == "Moderate"

    def test_all_weak_chunks(self):
        """All weak chunks should yield CCS = 20."""
        ccs = compute_concept_coverage_score(
            chunks_strong=0,
            chunks_moderate=0,
            chunks_weak=3,
            chunks_off_topic=0,
        )
        # (3 * 0.2) / 3 * 100 = 20
        assert ccs.score == pytest.approx(20.0)
        assert ccs.interpretation == "Low"

    def test_mixed_chunks_calculation(self):
        """Test CCS calculation with mixed bucket counts."""
        # 1 strong, 1 moderate, 1 weak, 1 off-topic
        ccs = compute_concept_coverage_score(
            chunks_strong=1,
            chunks_moderate=1,
            chunks_weak=1,
            chunks_off_topic=1,
        )
        # Weighted sum: 1*1.0 + 1*0.6 + 1*0.2 + 1*0.0 = 1.8
        # Score: (1.8 / 4) * 100 = 45
        assert ccs.weighted_sum == pytest.approx(1.8)
        assert ccs.score == pytest.approx(45.0)
        assert ccs.interpretation == "Weak"

    def test_bucket_counts_preserved(self):
        """CCS should preserve bucket counts."""
        ccs = compute_concept_coverage_score(
            chunks_strong=2,
            chunks_moderate=3,
            chunks_weak=4,
            chunks_off_topic=1,
        )
        assert ccs.bucket_counts["strong"] == 2
        assert ccs.bucket_counts["moderate"] == 3
        assert ccs.bucket_counts["weak"] == 4
        assert ccs.bucket_counts["off_topic"] == 1

    def test_total_chunks_computed(self):
        """Total chunks should be sum of all buckets."""
        ccs = compute_concept_coverage_score(
            chunks_strong=2,
            chunks_moderate=3,
            chunks_weak=4,
            chunks_off_topic=1,
        )
        assert ccs.total_chunks == 10

    def test_single_chunk_flag_true(self):
        """is_single_chunk should be True when total = 1."""
        ccs = compute_concept_coverage_score(
            chunks_strong=1,
            chunks_moderate=0,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        assert ccs.is_single_chunk is True

    def test_single_chunk_flag_false(self):
        """is_single_chunk should be False when total > 1."""
        ccs = compute_concept_coverage_score(
            chunks_strong=1,
            chunks_moderate=1,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        assert ccs.is_single_chunk is False

    def test_empty_chunks(self):
        """Empty chunks should yield CCS = 0."""
        ccs = compute_concept_coverage_score(
            chunks_strong=0,
            chunks_moderate=0,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        assert ccs.score == 0.0
        assert ccs.total_chunks == 0
        assert ccs.is_single_chunk is False

    def test_score_rounded_property(self):
        """score_rounded should return nearest integer."""
        ccs = compute_concept_coverage_score(
            chunks_strong=1,
            chunks_moderate=1,
            chunks_weak=1,
            chunks_off_topic=1,
        )
        # Score = 45.0
        assert ccs.score_rounded == 45

    def test_bucket_weights_reference(self):
        """CCS should include reference to weights used."""
        ccs = compute_concept_coverage_score(
            chunks_strong=1,
            chunks_moderate=0,
            chunks_weak=0,
            chunks_off_topic=0,
        )
        assert ccs.bucket_weights == COVERAGE_WEIGHTS


class TestInterpretCoverageScore:
    """Tests for interpret_coverage_score function."""

    def test_strong_interpretation(self):
        """Scores >= 80 should be 'Strong'."""
        assert interpret_coverage_score(100) == "Strong"
        assert interpret_coverage_score(80) == "Strong"
        assert interpret_coverage_score(95.5) == "Strong"

    def test_moderate_interpretation(self):
        """Scores 60-79 should be 'Moderate'."""
        assert interpret_coverage_score(79.9) == "Moderate"
        assert interpret_coverage_score(60) == "Moderate"
        assert interpret_coverage_score(70) == "Moderate"

    def test_weak_interpretation(self):
        """Scores 40-59 should be 'Weak'."""
        assert interpret_coverage_score(59.9) == "Weak"
        assert interpret_coverage_score(40) == "Weak"
        assert interpret_coverage_score(50) == "Weak"

    def test_low_interpretation(self):
        """Scores < 40 should be 'Low'."""
        assert interpret_coverage_score(39.9) == "Low"
        assert interpret_coverage_score(0) == "Low"
        assert interpret_coverage_score(20) == "Low"


class TestCoverageWeights:
    """Tests for coverage weight constants."""

    def test_weights_sum_to_expected(self):
        """Strong weight should be 1.0 (maximum)."""
        assert COVERAGE_WEIGHTS["strong"] == 1.0

    def test_moderate_weight(self):
        """Moderate weight should be 0.6."""
        assert COVERAGE_WEIGHTS["moderate"] == 0.6

    def test_weak_weight(self):
        """Weak weight should be 0.2."""
        assert COVERAGE_WEIGHTS["weak"] == 0.2

    def test_off_topic_weight(self):
        """Off-topic weight should be 0.0."""
        assert COVERAGE_WEIGHTS["off_topic"] == 0.0


class TestCoverageInterpretationThresholds:
    """Tests for coverage interpretation threshold constants."""

    def test_strong_threshold(self):
        """Strong threshold should be 80."""
        assert COVERAGE_INTERPRETATION["strong"] == 80

    def test_moderate_threshold(self):
        """Moderate threshold should be 60."""
        assert COVERAGE_INTERPRETATION["moderate"] == 60

    def test_weak_threshold(self):
        """Weak threshold should be 40."""
        assert COVERAGE_INTERPRETATION["weak"] == 40


class TestDiagnosticReportCoverage:
    """Tests for CCS integration in DiagnosticReport."""

    def test_report_has_coverage(self, sample_result):
        """DiagnosticReport should have coverage field."""
        report = create_diagnostic_report(sample_result)
        assert hasattr(report, 'coverage')
        assert isinstance(report.coverage, ConceptCoverageScore)

    def test_coverage_matches_summary_buckets(self, sample_result):
        """Coverage bucket counts should match summary counts."""
        report = create_diagnostic_report(sample_result)
        summary = report.summary
        coverage = report.coverage

        assert coverage.bucket_counts["strong"] == summary.chunks_strong
        assert coverage.bucket_counts["moderate"] == summary.chunks_moderate
        assert coverage.bucket_counts["weak"] == summary.chunks_weak
        assert coverage.bucket_counts["off_topic"] == summary.chunks_off_topic

    def test_coverage_total_matches_summary(self, sample_result):
        """Coverage total chunks should match summary total."""
        report = create_diagnostic_report(sample_result)
        assert report.coverage.total_chunks == report.summary.total_chunks

    def test_sample_result_coverage_score(self, sample_result):
        """Sample result should have expected CCS."""
        report = create_diagnostic_report(sample_result)
        # 1 strong (1.0), 1 moderate (0.6), 1 weak (0.2), 1 off-topic (0.0)
        # Weighted sum: 1.0 + 0.6 + 0.2 + 0.0 = 1.8
        # Score: (1.8 / 4) * 100 = 45
        assert report.coverage.score == pytest.approx(45.0)
        assert report.coverage.interpretation == "Weak"

    def test_uniform_moderate_coverage(self, uniform_result):
        """Uniform moderate scores should yield CCS = 60."""
        report = create_diagnostic_report(uniform_result)
        # All 3 chunks are moderate (0.70)
        # Weighted sum: 3 * 0.6 = 1.8
        # Score: (1.8 / 3) * 100 = 60
        assert report.coverage.score == pytest.approx(60.0)
        assert report.coverage.interpretation == "Moderate"

    def test_single_chunk_coverage_warning(self, single_chunk_result):
        """Single chunk document should have is_single_chunk flag."""
        report = create_diagnostic_report(single_chunk_result)
        assert report.coverage.is_single_chunk is True
