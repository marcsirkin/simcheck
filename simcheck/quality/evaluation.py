"""
Golden-set evaluation for the Page Quality classifier (Feature 11).

Answers "how often does SimCheck agree with a human QRG rater?" for three
backends: Jev only, Claude only, and Hybrid (Jev + Claude on doubt).

Each backend is called ONCE per page and its answers cached. Hybrid at any
escalation threshold is then simulated offline by merging cached Jev and
Claude answers, so a threshold sweep costs nothing. (Approximation: live
Hybrid asks Claude only the escalated subset; here Claude answered the full
rubric. Answers to the same question are expected to be close.)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

from simcheck.quality.classifier import Answer, needs_escalation
from simcheck.quality.page_quality import PageQualityRating, rate_page
from simcheck.quality.rubric import PQ_LEVELS, PQ_SLIDER, QUESTIONS_BY_ID, QuestionKind
from simcheck.quality.snapshot import PageSnapshot


# Rough per-page Claude cost when escalating (USD), from live runs
CLAUDE_COST_PER_ESCALATED_PAGE = 0.011


@dataclass(frozen=True)
class GoldenLabel:
    """One human rating from eval/data/qrg_golden.csv."""
    url: str
    level: float          # 0-4 in half steps (from the 9-point slider)
    ymyl: Optional[str]
    purpose: Optional[str]
    trust: Optional[float]
    notes: str = ""


class EvaluationError(Exception):
    """Raised for malformed golden-set files or cached answers."""


def slider_to_level(label: str) -> float:
    """
    Convert a 9-point slider label ("Medium+") to a 0-4 level (2.5).

    Raises:
        EvaluationError: If the label is not on the slider
    """
    cleaned = label.strip()
    for i, name in enumerate(PQ_SLIDER):
        if name.lower() == cleaned.lower():
            return i / 2
    raise EvaluationError(f"Unknown Page Quality label {label!r}; expected one of {', '.join(PQ_SLIDER)}")


def load_golden(path: Path) -> list:
    """
    Load rated, included rows from the golden CSV.

    Rows with include != yes or an empty page_quality are skipped (not yet
    rated). Malformed values raise.

    Raises:
        EvaluationError: On unknown labels
    """
    labels = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("include") or "").strip().lower() != "yes":
                continue
            pq = (row.get("page_quality") or "").strip()
            if not pq:
                continue
            trust_raw = (row.get("trust") or "").strip()
            trust = None
            if trust_raw:
                matches = [i for i, name in enumerate(PQ_LEVELS) if name.lower() == trust_raw.lower()]
                if not matches:
                    raise EvaluationError(f"Unknown trust label {trust_raw!r} for {row['url']}")
                trust = float(matches[0])
            labels.append(GoldenLabel(
                url=row["url"].strip(),
                level=slider_to_level(pq),
                ymyl=(row.get("ymyl") or "").strip().lower() or None,
                purpose=(row.get("purpose") or "").strip().lower() or None,
                trust=trust,
                notes=(row.get("notes") or "").strip(),
            ))
    return labels


# =============================================================================
# Answer caching
# =============================================================================

def answers_to_json(answers: dict) -> dict:
    """Serialize {id: Answer} for caching."""
    out = {}
    for qid, a in answers.items():
        d = asdict(a)
        d["kind"] = a.kind.value
        d["probabilities"] = {str(k): v for k, v in a.probabilities.items()}
        out[qid] = d
    return out


def answers_from_json(data: dict) -> dict:
    """
    Deserialize cached answers.

    Raises:
        EvaluationError: On unknown question IDs
    """
    answers = {}
    for qid, d in data.items():
        if qid not in QUESTIONS_BY_ID:
            raise EvaluationError(f"Cached answer for unknown question {qid!r} (rubric changed?)")
        kind = QuestionKind(d["kind"])
        probs = d.get("probabilities") or {}
        if kind == QuestionKind.SCORE:
            probs = {int(k): v for k, v in probs.items()}
        answers[qid] = Answer(qid, kind, d["value"], probs, d["confidence"], d.get("backend", ""),
                              d.get("evidence"))
    return answers


class ReplayClassifier:
    """Returns pre-computed answers; lets rate_page run without API calls."""

    def __init__(self, answers: dict, name: str = "replay"):
        self._answers = answers
        self.name = name

    def classify(self, state: dict, questions: list) -> dict:
        return {q.id: self._answers[q.id] for q in questions if q.id in self._answers}


def simulate_hybrid(jev: dict, claude: dict, threshold: float) -> tuple:
    """
    Merge cached answers as live Hybrid would at a given threshold.

    Returns:
        (merged answers, escalated question IDs)
    """
    escalated = needs_escalation(jev, threshold)
    merged = dict(jev)
    for qid in escalated:
        if qid in claude:
            merged[qid] = claude[qid]
    return merged, escalated


def rate_with_answers(snapshot: PageSnapshot, answers: dict) -> PageQualityRating:
    """Run the full rating (gates, blend) over cached answers."""
    return rate_page(snapshot, ReplayClassifier(answers))


# =============================================================================
# Metrics
# =============================================================================

@dataclass(frozen=True)
class AgreementMetrics:
    """Agreement of predicted ratings with golden labels."""
    n: int
    exact_slider: float      # same 9-point position
    within_half: float       # within one slider step (0.5 level)
    within_one: float        # within one full level (QRG "close enough")
    mae_levels: float        # mean absolute error, 0-4 scale
    mean_bias: float         # + = tool rates higher than human
    ymyl_agreement: Optional[float]
    purpose_agreement: Optional[float]
    trust_mae: Optional[float]


def _rate(hits: list) -> Optional[float]:
    return mean(hits) if hits else None


def compute_metrics(pairs: list) -> AgreementMetrics:
    """
    Compute agreement over (GoldenLabel, PageQualityRating) pairs.

    Unrated predictions are excluded from level metrics (they're
    crawlability findings, not quality calls).

    Raises:
        EvaluationError: If no rated pairs remain
    """
    rated = [(g, r) for g, r in pairs if r.rated]
    if not rated:
        raise EvaluationError("No rated predictions to evaluate.")
    # Compare on the slider grid so tiny fractional differences don't count
    errors = [round(r.level * 2) / 2 - g.level for g, r in rated]
    ymyl = [g.ymyl == r.ymyl for g, r in rated if g.ymyl]
    purpose = [g.purpose == r.purpose for g, r in rated if g.purpose]
    trust = [abs(r.eeat["trust"] - g.trust) for g, r in rated if g.trust is not None]
    return AgreementMetrics(
        n=len(rated),
        exact_slider=mean(abs(e) < 0.25 for e in errors),
        within_half=mean(abs(e) <= 0.5 for e in errors),
        within_one=mean(abs(e) <= 1.0 for e in errors),
        mae_levels=mean(abs(e) for e in errors),
        mean_bias=mean(errors),
        ymyl_agreement=_rate(ymyl),
        purpose_agreement=_rate(purpose),
        trust_mae=mean(trust) if trust else None,
    )


def band_changes(base: list, other: list) -> dict:
    """
    How often a second backend changes the rating (no labels needed).

    Args:
        base, other: aligned lists of PageQualityRating for the same pages

    Returns:
        {"pages", "slider_changed", "level_changed_1plus", "mean_abs_shift", "mean_shift"}
    """
    pairs = [(a, b) for a, b in zip(base, other) if a.rated and b.rated]
    shifts = [b.level - a.level for a, b in pairs]
    return {
        "pages": len(pairs),
        "slider_changed": sum(a.band != b.band for a, b in pairs),
        "level_changed_1plus": sum(abs(s) >= 1.0 for s in shifts),
        "mean_abs_shift": mean(abs(s) for s in shifts) if shifts else 0.0,
        "mean_shift": mean(shifts) if shifts else 0.0,
    }


def write_cache(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL cache."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_cache(path: Path) -> dict:
    """Load a JSONL cache keyed by (url, backend, rubric_version)."""
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[(rec["url"], rec["backend"], rec["rubric_version"])] = rec
    return cache
