"""
Google Search Quality Rater Guidelines rubric, expressed as typed questions.

Each RubricQuestion maps 1:1 to a Jev question type (Choice / Score / Noul)
and to a field in the Claude JSON schema, so both backends answer the same
rubric. Wording is paraphrased from the QRG (edition in RUBRIC_VERSION);
section numbers are noted for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Bump the suffix when question wording or criteria change, so stored
# results are only compared against results from the same rubric.
QRG_EDITION = "2025-09-11"
RUBRIC_VERSION = f"qrg-{QRG_EDITION}.v2"

# QRG Page Quality levels, index = numeric level (0-4)
PQ_LEVELS = ("Lowest", "Low", "Medium", "High", "Highest")

# QRG's 9-point slider: whole levels plus in-between "+" positions
PQ_SLIDER = ("Lowest", "Lowest+", "Low", "Low+", "Medium",
             "Medium+", "High", "High+", "Highest")

NEEDS_MET_LEVELS = ("Fails to Meet", "Slightly Meets", "Moderately Meets",
                    "Highly Meets", "Fully Meets")


class QuestionKind(Enum):
    CHOICE = "choice"  # unordered labels
    SCORE = "score"    # ordered levels 0..n-1
    NOUL = "noul"      # yes/no probability


@dataclass(frozen=True)
class RubricQuestion:
    """
    One typed rubric question.

    criteria: tuple of ordered level descriptions (SCORE), tuple of
    (label, description) pairs (CHOICE), or None (NOUL).
    """
    id: str
    kind: QuestionKind
    instructions: str
    criteria: Optional[tuple] = None
    requires_query: bool = False
    qrg_section: str = ""

    def labels(self) -> tuple:
        """Choice labels in order (CHOICE only)."""
        return tuple(label for label, _ in self.criteria) if self.kind == QuestionKind.CHOICE else ()


def _eeat_levels(dimension: str) -> tuple:
    return (
        f"Lowest: no {dimension}, or evidence against it",
        f"Low: little {dimension} for this topic",
        f"Medium: adequate {dimension} for this topic",
        f"High: clear, demonstrated {dimension}",
        f"Highest: exceptional, widely recognized {dimension}",
    )


QUESTIONS = (
    RubricQuestion(
        id="page_purpose",
        kind=QuestionKind.CHOICE,
        instructions="What is the primary purpose of this page?",
        criteria=(
            ("informational", "Share information or explain a topic"),
            ("commercial", "Promote, review, or compare products or services"),
            ("transactional", "Let the user buy, sign up, donate, or download"),
            ("navigational", "Homepage or hub that routes users to other pages"),
            ("entertainment", "Entertain, share media, or express opinion"),
            ("none_or_harmful", "No beneficial purpose, or created to deceive or harm"),
        ),
        qrg_section="2.2",
    ),
    RubricQuestion(
        id="ymyl",
        kind=QuestionKind.CHOICE,
        instructions=("Is this a Your Money or Your Life (YMYL) topic: could inaccurate content "
                      "significantly harm a person's health, financial stability, safety, or "
                      "society?"),
        criteria=(
            ("no", "Not YMYL: little potential for harm"),
            ("possibly", "Some potential for harm depending on the details"),
            ("clearly", "Clear potential for significant harm"),
        ),
        qrg_section="2.3",
    ),
    RubricQuestion(
        id="ymyl_topic",
        kind=QuestionKind.CHOICE,
        instructions="If any part of this page could cause harm when inaccurate, which area is it?",
        criteria=(
            ("health", "Medical, mental health, nutrition, drugs"),
            ("finance", "Money, investing, taxes, insurance, loans, purchases"),
            ("safety", "Physical safety, legal, emergency"),
            ("society", "Civic, news, groups of people, public trust"),
            ("none", "Not applicable"),
        ),
        qrg_section="2.3",
    ),
    RubricQuestion(
        id="mc_quality",
        kind=QuestionKind.SCORE,
        instructions=("How good is the page's main content, judged by effort, originality, "
                      "talent or skill, and accuracy for its purpose?"),
        criteria=(
            "Lowest: no main content, copied, auto-generated, or no effort",
            "Low: little effort or originality, thin, or exaggerated",
            "Medium: adequate effort; achieves its purpose but nothing special",
            "High: significant effort, originality, and skill",
            "Highest: exceptional effort, originality, talent, or skill",
        ),
        qrg_section="3.2, 5.2, 7.1",
    ),
    RubricQuestion(
        id="experience",
        kind=QuestionKind.SCORE,
        instructions=("How much first-hand or life experience does the content creator show "
                      "for this topic (actually used the product, visited the place, lived it)? "
                      "Per QRG 3.4, experience matters only when the topic calls for it: for "
                      "reference, institutional, or organizational pages where first-hand "
                      "experience is not expected, rate Medium."),
        criteria=(
            "Lowest: claims experience falsely, or the topic needs experience and there is none",
            "Low: the topic needs first-hand experience and little is shown",
            "Medium: adequate experience, OR first-hand experience is not relevant to this page",
            "High: clear, demonstrated first-hand experience",
            "Highest: extensive first-hand experience that makes the content uniquely valuable",
        ),
        qrg_section="3.4",
    ),
    RubricQuestion(
        id="expertise",
        kind=QuestionKind.SCORE,
        instructions="How much knowledge or skill does the content creator show for this topic?",
        criteria=_eeat_levels("expertise"),
        qrg_section="3.4",
    ),
    RubricQuestion(
        id="authoritativeness",
        kind=QuestionKind.SCORE,
        instructions=("How well known is the website or creator as a go-to source for this "
                      "topic?"),
        criteria=_eeat_levels("authoritativeness"),
        qrg_section="3.4",
    ),
    RubricQuestion(
        id="trust",
        kind=QuestionKind.SCORE,
        instructions=("How trustworthy is this page: accurate, honest, safe, and transparent "
                      "about who is responsible for it? Trust is the most important part of "
                      "E-E-A-T."),
        criteria=_eeat_levels("trustworthiness"),
        qrg_section="3.4",
    ),
    RubricQuestion(
        id="reputation_evident",
        kind=QuestionKind.NOUL,
        instructions=("Does the page or its signals show a positive reputation for the website "
                      "or content creator (awards, credentials, recognized brand, independent "
                      "references)?"),
        qrg_section="3.3",
    ),
    RubricQuestion(
        id="ads_obstruct",
        kind=QuestionKind.NOUL,
        instructions=("Do ads, affiliate links, or other monetization obstruct, dominate, or "
                      "distract from the main content?"),
        qrg_section="2.4.3, 5.0",
    ),
    RubricQuestion(
        id="deceptive_or_harmful",
        kind=QuestionKind.NOUL,
        instructions=("Is this page deceptive, harmful, untrustworthy, or created mainly to "
                      "manipulate users or search engines?"),
        qrg_section="4.0",
    ),
    RubricQuestion(
        id="low_effort_scaled",
        kind=QuestionKind.NOUL,
        instructions=("Does the main content look mass-produced or auto-generated (for example "
                      "low-effort AI text) with little original value?"),
        qrg_section="4.6.5",
    ),
    RubricQuestion(
        id="purpose_achieved",
        kind=QuestionKind.NOUL,
        instructions="Does the page achieve its purpose well for a typical visitor?",
        qrg_section="3.1",
    ),
    RubricQuestion(
        id="page_quality",
        kind=QuestionKind.SCORE,
        instructions=("Considering purpose, harm potential, main content quality, reputation, "
                      "and E-E-A-T, what overall Page Quality rating does this page deserve "
                      "under Google's Search Quality Rater Guidelines?"),
        criteria=(
            "Lowest: untrustworthy, deceptive, harmful, or no beneficial purpose",
            "Low: lacking E-E-A-T, low-effort main content, or distracting ads",
            "Medium: achieves its purpose; nothing wrong but nothing special",
            "High: satisfying main content, clear E-E-A-T, positive reputation",
            "Highest: very high E-E-A-T, exceptional effort, excellent reputation",
        ),
        qrg_section="3.1, 4.0-8.0",
    ),
    RubricQuestion(
        id="needs_met",
        kind=QuestionKind.SCORE,
        instructions=("How well does this page meet the needs of a user searching for the "
                      "query in the state's 'query' field?"),
        criteria=(
            "Fails to Meet: unhelpful or off-topic for the query",
            "Slightly Meets: helpful for few users; low quality or weak match",
            "Moderately Meets: helpful for many users or somewhat helpful for most",
            "Highly Meets: very helpful for most users",
            "Fully Meets: the complete answer; nearly all users satisfied",
        ),
        requires_query=True,
        qrg_section="13.0",
    ),
)

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}


def questions_for(query: Optional[str] = None) -> list:
    """
    Rubric questions applicable to a rating request.

    Args:
        query: Target query; Needs Met is only asked when one is given

    Returns:
        List of RubricQuestion
    """
    has_query = bool(query and query.strip())
    return [q for q in QUESTIONS if has_query or not q.requires_query]


def slider_label(level: float) -> str:
    """
    Map a 0-4 level to QRG's 9-point slider label (nearest half step).

    Args:
        level: Expected Page Quality level (0.0-4.0)

    Returns:
        Label such as "Medium+" or "High"
    """
    clamped = min(max(level, 0.0), 4.0)
    # int(x + 0.5) rounds halves up; round() would use banker's rounding
    return PQ_SLIDER[int(clamped * 2 + 0.5)]
