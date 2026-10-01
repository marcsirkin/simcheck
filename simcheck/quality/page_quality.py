"""
Page Quality rating (Feature 10).

Combines classifier answers with deterministic QRG gates into a headline:
the QRG 9-point band (Lowest .. Highest) plus a 0-100 PQ score for
tracking change between drafts.

Gates come straight from the QRG and override the model:
- Deceptive/harmful (QRG 4.0) -> Lowest
- Mass-produced low-effort content (QRG 4.6.5) -> capped at Low
- Ads dominate the main content (QRG 5.0) -> capped at Low
- Clearly YMYL with no identifiable author AND no About/Contact
  information (QRG 2.5.2, 5.0) -> capped at Low

Pages with too little server-rendered content are returned unrated rather
than rated Lowest: the tool cannot see them, which is a crawlability
finding, not a quality judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from simcheck.quality.ai_access import AIAccessReport
from simcheck.quality.classifier import (
    Answer,
    ClassifierError,
    HybridClassifier,
    QualityClassifier,
    build_state,
)
from simcheck.quality.rubric import (
    NEEDS_MET_LEVELS,
    PQ_LEVELS,
    RUBRIC_VERSION,
    questions_for,
    slider_label,
)
from simcheck.quality.snapshot import PageSnapshot


# Fewer server-rendered words than this -> unrated
MIN_RATEABLE_WORDS = 50

# Gate thresholds on yes-probabilities (tune against the golden set)
DECEPTIVE_GATE = 0.7
LOW_EFFORT_GATE = 0.8
ADS_GATE = 0.7

LOWEST_LEVEL = 0.0
LOW_CAP_LEVEL = 1.0

# PQ score blend: holistic Page Quality answer vs E-E-A-T composite
PQ_BLEND_WEIGHT = 0.5
# E-E-A-T composite weights; Trust weighted highest per QRG 3.4
EEAT_WEIGHTS = {"trust": 0.4, "experience": 0.2, "expertise": 0.2, "authoritativeness": 0.2}

# PQ score interpretation bands (0-100)
PQ_SCORE_BANDS = ((80, "High"), (60, "Medium"), (40, "Low"))


@dataclass(frozen=True)
class PageQualityRating:
    """
    Page Quality result.

    level: final 0-4 Page Quality level after gates (None if unrated)
    band: QRG 9-point slider label, or "Unrated"
    pq_score: 0-100 blend of level and E-E-A-T (None if unrated)
    eeat: {dimension: 0-4 level}
    gates: messages for gates that changed the rating
    reasons: short human-readable drivers of the rating
    provenance: {question_id: backend name}
    """
    url: str
    rated: bool
    band: str
    level: Optional[float]
    pq_score: Optional[float]
    model_level: Optional[float]
    confidence: Optional[float]
    eeat: dict
    purpose: Optional[str]
    ymyl: Optional[str]
    ymyl_topic: Optional[str]
    needs_met: Optional[str]
    gates: tuple
    reasons: tuple
    provenance: dict
    escalated: tuple
    escalation_error: Optional[str]
    rubric_version: str
    answers: dict = field(default_factory=dict, repr=False)

    @property
    def pq_score_rounded(self) -> Optional[int]:
        """PQ score rounded for display."""
        return None if self.pq_score is None else round(self.pq_score)


def _value(answers: dict, qid: str, default=None):
    a: Optional[Answer] = answers.get(qid)
    return a.value if a is not None else default


def _eeat_composite(eeat: dict) -> float:
    """Weighted E-E-A-T level (0-4)."""
    return sum(EEAT_WEIGHTS[k] * eeat[k] for k in EEAT_WEIGHTS)


def _apply_gates(level: float, answers: dict, snapshot: PageSnapshot) -> tuple:
    """Return (gated level, gate messages)."""
    gates = []
    if _value(answers, "deceptive_or_harmful", 0.0) > DECEPTIVE_GATE:
        gates.append("Deceptive or harmful signals: rated Lowest (QRG 4.0).")
        return LOWEST_LEVEL, gates

    cap = None
    if _value(answers, "low_effort_scaled", 0.0) > LOW_EFFORT_GATE:
        gates.append("Main content looks mass-produced or low-effort: capped at Low (QRG 4.6.5).")
        cap = LOW_CAP_LEVEL
    if _value(answers, "ads_obstruct", 0.0) > ADS_GATE:
        gates.append("Ads or affiliate links dominate the main content: capped at Low (QRG 5.0).")
        cap = LOW_CAP_LEVEL
    rep = snapshot.reputation
    if (_value(answers, "ymyl") == "clearly" and not snapshot.author_name
            and not rep.about and not rep.contact):
        gates.append("YMYL topic with no identifiable author or About/Contact information: "
                     "capped at Low (QRG 2.5.2).")
        cap = LOW_CAP_LEVEL

    return (min(level, cap) if cap is not None else level), gates


def _reasons(answers: dict, snapshot: PageSnapshot, eeat: dict) -> list:
    """Short drivers of the rating, most important first."""
    reasons = []
    ymyl = _value(answers, "ymyl")
    if ymyl == "clearly":
        topic = _value(answers, "ymyl_topic")
        topic_text = f" ({topic})" if topic and topic != "none" else ""
        reasons.append(f"Clearly YMYL{topic_text}: held to the highest E-E-A-T standard.")
    weakest = min(eeat, key=eeat.get)
    strongest = max(eeat, key=eeat.get)
    reasons.append(f"Trust rated {PQ_LEVELS[int(round(eeat['trust']))]}.")
    if eeat[weakest] < 2.0:
        reasons.append(f"Weakest E-E-A-T dimension: {weakest} "
                       f"({PQ_LEVELS[int(round(eeat[weakest]))]}).")
    elif eeat[strongest] >= 3.0 and strongest != "trust":
        reasons.append(f"Strongest E-E-A-T dimension: {strongest}.")
    if not snapshot.author_name:
        reasons.append("No author identified on the page or in schema.")
    if snapshot.reputation.found_count() <= 1:
        reasons.append("Few About/Contact/policy pages linked: hard to tell who is responsible.")
    if snapshot.external_link_count == 0 and snapshot.word_count >= 300:
        reasons.append("No outbound citations to sources.")
    if _value(answers, "purpose_achieved", 1.0) < 0.5:
        reasons.append("Page may not achieve its purpose well.")
    return reasons


def _unrated(snapshot: PageSnapshot, reason: str) -> PageQualityRating:
    return PageQualityRating(
        url=snapshot.final_url, rated=False, band="Unrated", level=None, pq_score=None,
        model_level=None, confidence=None, eeat={}, purpose=None, ymyl=None, ymyl_topic=None,
        needs_met=None, gates=(), reasons=(reason,), provenance={}, escalated=(),
        escalation_error=None, rubric_version=RUBRIC_VERSION,
    )


def pq_score_interpretation(score: float) -> str:
    """Plain-language band for a 0-100 PQ score."""
    for threshold, label in PQ_SCORE_BANDS:
        if score >= threshold:
            return label
    return "Lowest"


def rate_page(
    snapshot: PageSnapshot,
    classifier: QualityClassifier,
    access: Optional[AIAccessReport] = None,
    query: Optional[str] = None,
) -> PageQualityRating:
    """
    Rate a page against the QRG rubric.

    Args:
        snapshot: PageSnapshot of the page
        classifier: Backend (Jev, Claude, Hybrid, or Fake)
        access: Optional AIAccessReport (adds crawler facts to the state)
        query: Optional target query (adds Needs Met)

    Returns:
        PageQualityRating (rated=False when the page has too little
        server-rendered content to judge)

    Raises:
        ClassifierError: If the classifier fails
    """
    if snapshot.word_count < MIN_RATEABLE_WORDS:
        return _unrated(
            snapshot,
            f"Only {snapshot.word_count} words of server-rendered content. The page is likely "
            "built by JavaScript or blocked; crawlers that don't run JavaScript see the same. "
            "Not rated: this is a crawlability problem, not a quality judgment.",
        )

    questions = questions_for(query)
    answers = classifier.classify(build_state(snapshot, access, query), questions)

    missing = [q.id for q in questions if q.id not in answers]
    if missing:
        raise ClassifierError(f"Classifier did not answer: {', '.join(missing)}")

    eeat = {k: float(answers[k].value) for k in EEAT_WEIGHTS}
    model_level = float(answers["page_quality"].value)
    level, gates = _apply_gates(model_level, answers, snapshot)

    if gates:
        # Gates cap the whole score, not just the band: a capped page should
        # not show a High-looking number because its E-E-A-T answers were generous.
        blended = level
    else:
        # E-E-A-T can lift the score at most one level above the holistic
        # Page Quality answer, so one generous dimension can't dominate.
        eeat_level = min(_eeat_composite(eeat), level + 1.0)
        blended = PQ_BLEND_WEIGHT * level + (1 - PQ_BLEND_WEIGHT) * eeat_level
    pq_score = 100.0 * blended / 4.0

    needs_met = None
    if "needs_met" in answers:
        needs_met = NEEDS_MET_LEVELS[int(round(float(answers["needs_met"].value)))]

    escalated = tuple(classifier.last_escalated) if isinstance(classifier, HybridClassifier) else ()
    escalation_error = classifier.last_escalation_error if isinstance(classifier, HybridClassifier) else None

    return PageQualityRating(
        url=snapshot.final_url,
        rated=True,
        band=slider_label(level),
        level=level,
        pq_score=pq_score,
        model_level=model_level,
        confidence=answers["page_quality"].confidence,
        eeat=eeat,
        purpose=_value(answers, "page_purpose"),
        ymyl=_value(answers, "ymyl"),
        ymyl_topic=_value(answers, "ymyl_topic"),
        needs_met=needs_met,
        gates=tuple(gates),
        reasons=tuple(gates + _reasons(answers, snapshot, eeat)),
        provenance={qid: a.backend for qid, a in answers.items()},
        escalated=escalated,
        escalation_error=escalation_error,
        rubric_version=RUBRIC_VERSION,
        answers=answers,
    )
