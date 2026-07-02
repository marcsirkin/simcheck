"""
SimScore: LLM Readiness Score (Feature 8).

A single reportable 0-100 metric that answers "how ready is this page to be
cited or summarized by AI search?" It blends the semantic Concept Coverage
Score with the structural and evidence signals that correlate with AI
answerability, so improvements to either move the headline number.

Composition (weights in READINESS_WEIGHTS):
- coverage (50%): CCS — does the content semantically express the topic?
- structure (20%): headings, TL;DR, FAQ, steps (steps only required for how-to)
- evidence (15%): outbound links, sources section, examples, numeric specifics
- answerability (15%): definition near the top, intro names the topic,
  best-matching content appears early

This module does NOT recompute embeddings or similarity; it is a pure
transformation over DiagnosticReport + ContentSignals.
"""

from dataclasses import dataclass

from simcheck.core.diagnostics import DiagnosticReport
from simcheck.core.geo import ContentSignals, GeoIntent


# Component weights; must sum to 1.0
READINESS_WEIGHTS = {
    "coverage": 0.50,
    "structure": 0.20,
    "evidence": 0.15,
    "answerability": 0.15,
}

# Interpretation bands for the composite score (0-100)
READINESS_INTERPRETATION = {
    "ready": 80,        # 80-100: AI-ready
    "nearly": 60,       # 60-79: Nearly ready
    "needs_work": 40,   # 40-59: Needs work
    # < 40: Not ready
}


@dataclass(frozen=True)
class ReadinessScore:
    """
    Composite LLM readiness score (SimScore).

    Attributes:
        score: Composite score (0-100)
        components: Per-component subscores (0-100), keyed by component name
        weights: Weights used to blend components
        interpretation: Human-readable band label
    """
    score: float
    components: dict
    weights: dict
    interpretation: str

    @property
    def score_rounded(self) -> int:
        """Score rounded to nearest integer for display."""
        return round(self.score)


def interpret_readiness(score: float) -> str:
    """
    Convert a readiness score to a human-readable interpretation.

    Args:
        score: SimScore value (0-100)

    Returns:
        Interpretation string
    """
    if score >= READINESS_INTERPRETATION["ready"]:
        return "AI-ready"
    elif score >= READINESS_INTERPRETATION["nearly"]:
        return "Nearly ready"
    elif score >= READINESS_INTERPRETATION["needs_work"]:
        return "Needs work"
    else:
        return "Not ready"


def _structure_component(signals: ContentSignals, intent: GeoIntent) -> float:
    """Score document structure signals (0-1)."""
    score = 0.0
    if signals.h2_count >= 2:
        score += 0.35
    elif signals.h2_count == 1:
        score += 0.15
    if signals.h3_count >= 1:
        score += 0.15
    if signals.has_tldr:
        score += 0.20
    if signals.has_faq:
        score += 0.15
    # Numbered steps only matter for how-to intent; other intents get
    # the credit unconditionally so they aren't penalized for a signal
    # that doesn't apply to them.
    if signals.has_steps or intent != GeoIntent.HOW_TO:
        score += 0.15
    return min(score, 1.0)


def _evidence_component(signals: ContentSignals) -> float:
    """Score evidence/citation signals (0-1)."""
    score = 0.0
    if signals.link_count >= 2:
        score += 0.40
    elif signals.link_count == 1:
        score += 0.20
    if signals.has_sources_section:
        score += 0.30
    if signals.has_examples:
        score += 0.20
    if signals.numeric_density >= 0.2:
        score += 0.10
    return min(score, 1.0)


def _answerability_component(report: DiagnosticReport, signals: ContentSignals) -> float:
    """Score front-loading / direct-answer signals (0-1)."""
    score = 0.0
    if signals.has_definition_near_top:
        score += 0.40
    score += 0.40 * signals.intro_query_term_coverage
    best = report.get_max_chunk()
    if best is not None and best.position_percent <= 0.4:
        score += 0.20
    return min(score, 1.0)


def compute_readiness_score(
    report: DiagnosticReport,
    signals: ContentSignals,
    intent: GeoIntent,
) -> ReadinessScore:
    """
    Compute the composite LLM readiness score (SimScore).

    Args:
        report: DiagnosticReport from create_diagnostic_report()
        signals: ContentSignals from extract_content_signals()
        intent: Resolved query intent (not AUTO)

    Returns:
        ReadinessScore with composite score, component breakdown, and band
    """
    components_unit = {
        "coverage": report.coverage.score / 100.0,
        "structure": _structure_component(signals, intent),
        "evidence": _evidence_component(signals),
        "answerability": _answerability_component(report, signals),
    }

    score = 100.0 * sum(
        READINESS_WEIGHTS[name] * value for name, value in components_unit.items()
    )

    return ReadinessScore(
        score=score,
        components={name: value * 100.0 for name, value in components_unit.items()},
        weights=READINESS_WEIGHTS,
        interpretation=interpret_readiness(score),
    )
