"""Experimental content-pattern score.

This score combines semantic coverage with observable structure, evidence,
and opening clarity.  It is an editorial diagnostic, not a prediction that an
AI engine will cite the page. Citation behavior is measured separately by the
live probes.

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
from simcheck.core.geo import ContentSignals, GeoIntent, PageType


# Component weights; must sum to 1.0
READINESS_WEIGHTS = {
    "coverage": 0.50,
    "structure": 0.20,
    "evidence": 0.15,
    "answerability": 0.15,
}

# Descriptive bands for this uncalibrated composite. They intentionally avoid
# "ready/not ready" language because the score does not predict citations.
READINESS_INTERPRETATION = {
    "strong": 80,
    "solid": 60,
    "partial": 40,
}


@dataclass(frozen=True)
class ReadinessScore:
    """
    Experimental content-pattern score.

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
        score: Content-pattern value (0-100)

    Returns:
        Interpretation string
    """
    if score >= READINESS_INTERPRETATION["strong"]:
        return "Strong signal coverage"
    elif score >= READINESS_INTERPRETATION["solid"]:
        return "Solid signal coverage"
    elif score >= READINESS_INTERPRETATION["partial"]:
        return "Partial signal coverage"
    else:
        return "Limited signal coverage"


def _structure_component(signals: ContentSignals, intent: GeoIntent, page_type: PageType) -> float:
    """Score only structure signals relevant to this kind of page (0-1)."""
    if page_type == PageType.HOMEPAGE:
        return min((0.7 if signals.h2_count >= 2 else 0.35 if signals.h2_count == 1 else 0.0)
                   + (0.3 if signals.h3_count else 0.0), 1.0)
    if page_type == PageType.COMMERCIAL:
        return min((0.4 if signals.h2_count >= 2 else 0.2 if signals.h2_count == 1 else 0.0)
                   + (0.15 if signals.h3_count else 0.0)
                   + (0.25 if signals.has_comparison_language else 0.0)
                   + (0.2 if signals.has_faq else 0.0), 1.0)

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


def _evidence_component(signals: ContentSignals, page_type: PageType) -> float:
    """Score page-appropriate evidence signals (0-1)."""
    if page_type in (PageType.HOMEPAGE, PageType.COMMERCIAL):
        return min(
            (0.35 if signals.has_examples else 0.0)
            + (0.30 if signals.numeric_density >= 0.2 else 0.0)
            + (0.20 if signals.link_count >= 2 else 0.10 if signals.link_count == 1 else 0.0)
            + (0.15 if signals.has_freshness_signals else 0.0),
            1.0,
        )
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


def _answerability_component(
    report: DiagnosticReport,
    signals: ContentSignals,
    page_type: PageType,
) -> float:
    """Score whether the opening communicates the page's topic or purpose."""
    score = 0.0
    definition_weight = 0.40 if page_type not in (PageType.HOMEPAGE, PageType.COMMERCIAL) else 0.0
    if definition_weight and signals.has_definition_near_top:
        score += definition_weight
        intro_weight = 0.40
    else:
        intro_weight = 0.40 if definition_weight else 0.70
    score += intro_weight * signals.intro_query_term_coverage
    best = report.get_max_chunk()
    if best is not None and best.position_percent <= 0.4:
        score += 1.0 - intro_weight - definition_weight
    return min(score, 1.0)


def compute_readiness_score(
    report: DiagnosticReport,
    signals: ContentSignals,
    intent: GeoIntent,
    page_type: PageType = PageType.ARTICLE,
) -> ReadinessScore:
    """
    Compute the experimental content-pattern score.

    Args:
        report: DiagnosticReport from create_diagnostic_report()
        signals: ContentSignals from extract_content_signals()
        intent: Resolved query intent (not AUTO)
        page_type: The job the page itself is meant to do

    Returns:
        ReadinessScore with composite score, component breakdown, and band
    """
    components_unit = {
        "coverage": report.coverage.score / 100.0,
        "structure": _structure_component(signals, intent, page_type),
        "evidence": _evidence_component(signals, page_type),
        "answerability": _answerability_component(report, signals, page_type),
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
