"""Tests for the SimScore LLM readiness metric (Feature 8)."""

from simcheck.core.models import Chunk, ChunkSimilarity, ComparisonResult
from simcheck.core.diagnostics import create_diagnostic_report
from simcheck.core.geo import ContentSignals, GeoIntent
from simcheck.core.readiness import (
    compute_readiness_score,
    interpret_readiness,
    READINESS_WEIGHTS,
)


def make_chunk_similarity(index: int, similarity: float, interpretation: str) -> ChunkSimilarity:
    chunk = Chunk(
        index=index,
        text=f"chunk {index} text",
        char_start=index * 100,
        char_end=(index * 100) + 13,
        token_count=10,
    )
    return ChunkSimilarity(chunk=chunk, similarity=similarity, interpretation=interpretation)


def make_report(similarities: list[tuple[float, str]], query: str = "test topic phrase"):
    chunk_sims = [
        make_chunk_similarity(i, sim, interp) for i, (sim, interp) in enumerate(similarities)
    ]
    scores = [cs.similarity for cs in chunk_sims]
    result = ComparisonResult(
        query=query,
        document_char_count=100,
        document_token_count=50,
        chunk_count=len(chunk_sims),
        max_similarity=max(scores),
        max_similarity_chunk_index=scores.index(max(scores)),
        avg_similarity=sum(scores) / len(scores),
        chunk_similarities=chunk_sims,
        model_name="test-model",
        embedding_dim=384,
    )
    return create_diagnostic_report(result)


def make_signals(**overrides) -> ContentSignals:
    defaults = dict(
        word_count=500,
        h2_count=0,
        h3_count=0,
        link_count=0,
        table_like_lines=0,
        has_faq=False,
        has_tldr=False,
        has_sources_section=False,
        has_steps=False,
        has_definition_near_top=False,
        has_examples=False,
        has_comparison_language=False,
        has_freshness_signals=False,
        numeric_density=0.0,
        intro_query_term_coverage=0.0,
    )
    defaults.update(overrides)
    return ContentSignals(**defaults)


class TestInterpretReadiness:
    def test_bands(self):
        assert interpret_readiness(85) == "AI-ready"
        assert interpret_readiness(80) == "AI-ready"
        assert interpret_readiness(70) == "Nearly ready"
        assert interpret_readiness(50) == "Needs work"
        assert interpret_readiness(20) == "Not ready"


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(READINESS_WEIGHTS.values()) - 1.0) < 1e-9


class TestComputeReadinessScore:
    def test_bare_document_scores_low(self):
        """No structure, no evidence, weak coverage -> low score."""
        report = make_report([(0.30, "Off-topic"), (0.35, "Off-topic")])
        score = compute_readiness_score(report, make_signals(), GeoIntent.INFORMATIONAL)
        assert score.score < 40
        assert score.interpretation == "Not ready"

    def test_fully_equipped_document_scores_high(self):
        """Strong coverage + all signals -> high score."""
        report = make_report([(0.90, "Strong"), (0.85, "Strong"), (0.88, "Strong")])
        signals = make_signals(
            h2_count=4,
            h3_count=2,
            link_count=3,
            has_faq=True,
            has_tldr=True,
            has_sources_section=True,
            has_examples=True,
            has_definition_near_top=True,
            numeric_density=0.4,
            intro_query_term_coverage=1.0,
        )
        score = compute_readiness_score(report, signals, GeoIntent.INFORMATIONAL)
        assert score.score >= 90
        assert score.interpretation == "AI-ready"

    def test_components_are_bounded(self):
        report = make_report([(0.90, "Strong")])
        signals = make_signals(
            h2_count=10, h3_count=10, link_count=10,
            has_faq=True, has_tldr=True, has_sources_section=True,
            has_examples=True, has_definition_near_top=True,
            numeric_density=1.0, intro_query_term_coverage=1.0,
        )
        score = compute_readiness_score(report, signals, GeoIntent.INFORMATIONAL)
        for name, value in score.components.items():
            assert 0.0 <= value <= 100.0, name
        assert 0.0 <= score.score <= 100.0

    def test_coverage_dominates(self):
        """Coverage carries half the weight: same signals, better coverage wins."""
        signals = make_signals(h2_count=3, has_tldr=True)
        weak = make_report([(0.30, "Off-topic"), (0.32, "Off-topic")])
        strong = make_report([(0.90, "Strong"), (0.88, "Strong")])
        weak_score = compute_readiness_score(weak, signals, GeoIntent.INFORMATIONAL)
        strong_score = compute_readiness_score(strong, signals, GeoIntent.INFORMATIONAL)
        assert strong_score.score - weak_score.score >= 40

    def test_steps_required_only_for_how_to(self):
        """Missing steps penalizes how-to intent but not informational."""
        report = make_report([(0.90, "Strong")])
        signals = make_signals(h2_count=2)
        info = compute_readiness_score(report, signals, GeoIntent.INFORMATIONAL)
        how_to = compute_readiness_score(report, signals, GeoIntent.HOW_TO)
        assert info.components["structure"] > how_to.components["structure"]

    def test_steps_close_the_how_to_gap(self):
        report = make_report([(0.90, "Strong")])
        with_steps = make_signals(h2_count=2, has_steps=True)
        info = compute_readiness_score(report, with_steps, GeoIntent.INFORMATIONAL)
        how_to = compute_readiness_score(report, with_steps, GeoIntent.HOW_TO)
        assert info.components["structure"] == how_to.components["structure"]

    def test_early_best_chunk_boosts_answerability(self):
        """Best-matching chunk early in the document raises answerability."""
        early_best = make_report([(0.90, "Strong"), (0.50, "Weak"), (0.50, "Weak")])
        late_best = make_report([(0.50, "Weak"), (0.50, "Weak"), (0.90, "Strong")])
        signals = make_signals()
        early = compute_readiness_score(early_best, signals, GeoIntent.INFORMATIONAL)
        late = compute_readiness_score(late_best, signals, GeoIntent.INFORMATIONAL)
        assert early.components["answerability"] > late.components["answerability"]
