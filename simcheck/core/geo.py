"""
GEO / AI-SEO oriented insights and next steps.

SimCheck's core score (CCS) measures semantic alignment between a target topic
and document chunks. For the primary use case of improving AI visibility
(a.k.a. GEO / AI SEO), users also need guidance that is:

- Interpretable: "Is this score good?"
- Actionable: "What should I change next?"
- Practical: "Where in the document should I start?"

This module adds a lightweight heuristic layer on top of DiagnosticReport,
plus simple, dependency-free content signals extracted from the raw document
text (Markdown/HTML-ish).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from simcheck.core.diagnostics import DiagnosticReport, ChunkDiagnostic
from simcheck.core.models import SIMILARITY_THRESHOLDS


class GeoIntent(Enum):
    AUTO = "auto"
    INFORMATIONAL = "informational"
    HOW_TO = "how_to"
    COMMERCIAL = "commercial"


class PageType(Enum):
    """The job the page itself is meant to do, separate from query intent."""

    AUTO = "auto"
    ARTICLE = "article"
    HOW_TO = "how_to"
    COMMERCIAL = "commercial"
    HOMEPAGE = "homepage"
    REFERENCE = "reference"


class GeoPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ContentSignals:
    """
    Observable structure and evidence signals used for editorial diagnostics.

    These are intentionally heuristic and format-agnostic.
    """
    word_count: int
    h2_count: int
    h3_count: int
    link_count: int
    table_like_lines: int
    has_faq: bool
    has_tldr: bool
    has_sources_section: bool
    has_steps: bool
    has_definition_near_top: bool
    has_examples: bool
    has_comparison_language: bool
    has_freshness_signals: bool
    numeric_density: float  # 0-1: fraction of lines containing numbers
    intro_query_term_coverage: float  # 0-1


@dataclass(frozen=True)
class GeoNextStep:
    title: str
    priority: GeoPriority
    why: str
    how: str
    minutes: int
    examples: Optional[str] = None
    target_chunks: Optional[List[ChunkDiagnostic]] = None


@dataclass(frozen=True)
class GeoNextStepsReport:
    summary: str
    steps: List[GeoNextStep]
    signals: ContentSignals
    intent: GeoIntent
    page_type: PageType = PageType.ARTICLE

    def high_priority(self) -> List[GeoNextStep]:
        return [s for s in self.steps if s.priority == GeoPriority.HIGH]

    def medium_priority(self) -> List[GeoNextStep]:
        return [s for s in self.steps if s.priority == GeoPriority.MEDIUM]

    def low_priority(self) -> List[GeoNextStep]:
        return [s for s in self.steps if s.priority == GeoPriority.LOW]


_CODE_FENCE_RE = re.compile(r"^\s*```")
_MD_H2_RE = re.compile(r"^\s*##\s+\S")
_MD_H3_RE = re.compile(r"^\s*###\s+\S")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
_PLAIN_URL_RE = re.compile(r"https?://[^\s)]+")
_TLDR_RE = re.compile(r"\bTL;?DR\b|\bSummary\b", re.IGNORECASE)
_FAQ_RE = re.compile(r"\bFAQ\b|frequently asked questions", re.IGNORECASE)
_SOURCES_RE = re.compile(r"^(references|sources|further reading)$", re.IGNORECASE)
_STEPS_RE = re.compile(r"^\s*(step\s+\d+|[0-9]+\.)\s+", re.IGNORECASE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_HTML_H2_RE = re.compile(r"<h2\b", re.IGNORECASE)
_HTML_H3_RE = re.compile(r"<h3\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"\b(is|are|refers to|means)\b", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"\b(for example|e\.g\.|example:)\b", re.IGNORECASE)
_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compare|comparison|alternatives?)\b", re.IGNORECASE)
_FRESHNESS_RE = re.compile(r"\b(updated|last updated|as of|new in)\b|\b20(1\d|2\d)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d")


def _iter_non_code_lines(document: str) -> Iterable[str]:
    """
    Iterate through lines, ignoring content inside Markdown code fences.

    This keeps basic structural heuristics from being polluted by code samples.
    """
    in_code_fence = False
    for line in document.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            continue
        if not in_code_fence:
            yield line


def _query_terms(query: str, max_terms: int = 8) -> List[str]:
    """
    Extract a few meaningful query terms for simple coverage checks.

    This is not NLP; it is just a pragmatic heuristic.
    """
    terms = []
    for raw in re.split(r"[^a-zA-Z0-9]+", query.lower()):
        if len(raw) < 4:
            continue
        if raw in {"with", "from", "that", "this", "your", "what", "when", "where", "which", "into"}:
            continue
        terms.append(raw)
    # preserve order but de-dup
    seen = set()
    unique = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        unique.append(t)
    return unique[:max_terms]


def infer_intent(query: str) -> GeoIntent:
    q = (query or "").strip().lower()
    if not q:
        return GeoIntent.INFORMATIONAL

    if re.search(r"\b(how to|steps?|tutorial|guide|checklist)\b", q):
        return GeoIntent.HOW_TO

    if re.search(r"\b(best|top|pricing|cost|cheap|review|tool|tools|software|platform|service|agency|template|alternatives?)\b", q):
        return GeoIntent.COMMERCIAL

    if re.search(r"\b(what is|definition|meaning)\b", q):
        return GeoIntent.INFORMATIONAL

    # Default: informational
    return GeoIntent.INFORMATIONAL


def infer_page_type(
    document: str,
    intent: GeoIntent,
    *,
    page_url: str = "",
    classified_purpose: Optional[str] = None,
) -> PageType:
    """Infer a conservative page type from available page context."""
    purpose = (classified_purpose or "").strip().lower()
    if purpose == "navigational":
        return PageType.HOMEPAGE
    if purpose in {"commercial", "transactional"}:
        return PageType.COMMERCIAL

    if page_url:
        path = urlparse(page_url).path.rstrip("/")
        if not path:
            return PageType.HOMEPAGE

    text = (document or "")[:3000].lower()
    if re.search(r"\b(api reference|developer reference|function reference|class reference)\b", text):
        return PageType.REFERENCE
    if intent == GeoIntent.HOW_TO:
        return PageType.HOW_TO
    if intent == GeoIntent.COMMERCIAL:
        return PageType.COMMERCIAL
    return PageType.ARTICLE


def extract_content_signals(document: str, query: str) -> ContentSignals:
    words = document.split()
    word_count = len(words)

    h2_count = 0
    h3_count = 0
    link_count = 0
    table_like_lines = 0
    has_faq = False
    has_sources_section = False
    has_steps = False
    has_definition_near_top = False
    has_examples = False
    has_comparison_language = False

    first_800_chars = document[:800]
    has_tldr = bool(_TLDR_RE.search(first_800_chars))
    has_freshness_signals = bool(_FRESHNESS_RE.search(document[:2000]))

    # Definition near top: very lightweight heuristic
    top_text = " ".join(words[:220])
    terms = _query_terms(query)
    if terms and _DEFINITION_RE.search(top_text):
        # Require at least one query term near the definition area
        has_definition_near_top = any(t in top_text.lower() for t in terms)

    plain_url_count = len(_PLAIN_URL_RE.findall(document))

    non_code_lines = list(_iter_non_code_lines(document))
    numeric_lines = 0

    for line in non_code_lines:
        if _MD_H2_RE.match(line):
            h2_count += 1
        if _MD_H3_RE.match(line):
            h3_count += 1
        link_count += len(_MD_LINK_RE.findall(line))
        if _TABLE_LINE_RE.match(line):
            table_like_lines += 1
        if _FAQ_RE.search(line):
            has_faq = True
        normalized_heading = re.sub(r"^\s{0,3}#+\s*", "", line).strip()
        if _SOURCES_RE.match(normalized_heading):
            has_sources_section = True
        if _STEPS_RE.match(line):
            has_steps = True
        if _EXAMPLE_RE.search(line):
            has_examples = True
        if _COMPARISON_RE.search(line):
            has_comparison_language = True
        if _NUMBER_RE.search(line):
            numeric_lines += 1

    # Add basic HTML heading detection (for pasted HTML)
    h2_count += len(_HTML_H2_RE.findall(document))
    h3_count += len(_HTML_H3_RE.findall(document))

    # Count plain URLs in addition to Markdown links, but avoid double-counting.
    link_count = max(link_count, plain_url_count)

    # Intro coverage: do we "name the thing" early?
    if not terms:
        intro_query_term_coverage = 0.0
    else:
        intro_text = " ".join(words[:200]).lower()
        covered = sum(1 for t in terms if t in intro_text)
        intro_query_term_coverage = covered / len(terms)

    return ContentSignals(
        word_count=word_count,
        h2_count=h2_count,
        h3_count=h3_count,
        link_count=link_count,
        table_like_lines=table_like_lines,
        has_faq=has_faq,
        has_tldr=has_tldr,
        has_sources_section=has_sources_section,
        has_steps=has_steps,
        has_definition_near_top=has_definition_near_top,
        has_examples=has_examples,
        has_comparison_language=has_comparison_language,
        has_freshness_signals=has_freshness_signals,
        numeric_density=(numeric_lines / max(len(non_code_lines), 1)),
        intro_query_term_coverage=intro_query_term_coverage,
    )


def _avg_similarity(chunks: List[ChunkDiagnostic]) -> float:
    if not chunks:
        return 0.0
    return sum(c.similarity for c in chunks) / len(chunks)


def generate_geo_next_steps(
    report: DiagnosticReport,
    document: str,
    *,
    intent_override: GeoIntent = GeoIntent.AUTO,
    page_type_override: PageType = PageType.AUTO,
    page_url: str = "",
    classified_purpose: Optional[str] = None,
    max_steps: int = 7,
) -> GeoNextStepsReport:
    """
    Generate GEO-oriented next steps based on the diagnostic report + raw text.

    This complements (not replaces) CCS recommendations, focusing on actions a
    content editor can take quickly: front-load answers, reduce drift, improve
    structure, and add evidence.
    """
    intent = infer_intent(report.query) if intent_override == GeoIntent.AUTO else intent_override
    page_type = (
        infer_page_type(document, intent, page_url=page_url, classified_purpose=classified_purpose)
        if page_type_override == PageType.AUTO
        else page_type_override
    )
    signals = extract_content_signals(document, report.query)

    total_chunks = report.summary.total_chunks
    if total_chunks <= 0:
        return GeoNextStepsReport(
            summary="No chunks found; add content and re-run analysis.",
            steps=[],
            signals=signals,
            intent=intent,
            page_type=page_type,
        )

    doc_chunks = report.by_document_order()
    first_band = [c for c in doc_chunks if c.position_percent <= 0.2]
    if len(first_band) < 2:
        first_band = doc_chunks[: min(3, len(doc_chunks))]

    intro_avg = _avg_similarity(first_band)
    overall_avg = report.summary.avg_similarity

    best = report.get_max_chunk()
    best_pos = best.position_percent if best else 0.0

    off_topic = report.off_topic_chunks()
    off_topic_percent = (len(off_topic) / total_chunks) if total_chunks else 0.0

    thresholds = report.effective_thresholds
    weak_threshold = thresholds["weak"]
    moderate_threshold = thresholds["moderate"]
    weak_chunks = [c for c in report.chunks if weak_threshold <= c.similarity < moderate_threshold]

    steps: List[GeoNextStep] = []

    # 1) Put the page's purpose early. Definitions are appropriate for
    # explanatory pages, not a universal requirement.
    front_load_needed = (
        signals.intro_query_term_coverage < 0.5
        or best_pos > 0.4
        or (intro_avg + 0.05) < overall_avg
    )
    if front_load_needed and page_type not in (PageType.HOMEPAGE, PageType.COMMERCIAL):
        target = [c for c in doc_chunks[: min(3, len(doc_chunks))]]
        steps.append(GeoNextStep(
            title="Consider clarifying the answer near the top",
            priority=GeoPriority.HIGH,
            minutes=15,
            why=(
                "The strongest-matching passage appears late or the introduction only partly names the target. "
                "A clearer opening may help readers and retrieval systems understand the page sooner."
            ),
            how=(
                "If it fits the page's voice, use a short lead that states the answer or outcome, who it is for, "
                "and what the page covers. Use the target language naturally; do not repeat it solely to raise the score."
            ),
            examples=(
                "Template:\n"
                "“<TOPIC> is <definition>. In this guide you’ll learn <3 key subtopics>. "
                "If you’re <audience/intent>, start with <first recommended action>.”"
            ),
            target_chunks=target,
        ))

    # 1b) Add a crisp definition near the top (especially informational queries)
    if page_type in (PageType.ARTICLE, PageType.REFERENCE) and intent == GeoIntent.INFORMATIONAL and not signals.has_definition_near_top:
        steps.append(GeoNextStep(
            title="Consider a concise definition near the top",
            priority=GeoPriority.HIGH if report.coverage.score < 70 else GeoPriority.MEDIUM,
            minutes=10,
            why="For explanatory and reference pages, a concise definition can reduce ambiguity for readers.",
            how=(
                "If the audience needs it, define the topic in one sentence and clarify its scope in a second. "
                "Skip this when a definition would make the opening feel mechanical."
            ),
        ))

    # 2) Remove or rewrite off-topic content (CCS + GEO)
    if off_topic_percent >= 0.10 and page_type != PageType.HOMEPAGE:
        priority = GeoPriority.HIGH if off_topic_percent >= 0.15 else GeoPriority.MEDIUM
        steps.append(GeoNextStep(
            title="Review lower-alignment sections for intentional drift",
            priority=priority,
            minutes=20,
            why=(
                "These sections have lower semantic alignment with the supplied target. They may be useful context, "
                "or they may be distracting; the score alone cannot decide which."
            ),
            how=(
                "Review each highlighted chunk. Keep it when it serves the page's purpose; otherwise connect it more "
                "clearly, condense it, move it to a better page, or remove it. Do not edit merely to cross a threshold."
            ),
            target_chunks=off_topic[:5],
        ))

    # 3) Strengthen weak chunks into "moderate"
    if weak_chunks:
        steps.append(GeoNextStep(
            title="Review partially aligned sections for clarity and specificity",
            priority=GeoPriority.MEDIUM,
            minutes=25,
            why=(
                "These chunks are semantically related but less aligned than stronger sections. That can reflect "
                "vagueness, intentional breadth, or a mismatch between the page and the target."
            ),
            how=(
                "Where useful, replace vague references with concrete entities, constraints, evidence, or examples. "
                "Preserve natural language and the author's voice rather than inserting target terms mechanically."
            ),
            target_chunks=weak_chunks[:5],
        ))

    # 3b) Intent-specific “answerability” structure
    if page_type == PageType.HOW_TO and not signals.has_steps:
        steps.append(GeoNextStep(
            title="Add step-by-step instructions (numbered steps + prerequisites)",
            priority=GeoPriority.HIGH if report.coverage.score < 70 else GeoPriority.MEDIUM,
            minutes=25,
            why="How-to queries perform better when the page contains explicit steps and prerequisites.",
            how=(
                "Add a `## Steps` section with 5–9 numbered steps. Start each step with an action verb. "
                "Add a short `## Prerequisites` list before the steps."
            ),
            examples="`## Prerequisites` ...\n\n`## Steps`\n1. ...\n2. ...",
        ))

    if page_type == PageType.COMMERCIAL and not signals.has_comparison_language and signals.word_count >= 400:
        steps.append(GeoNextStep(
            title="Add a comparison section (alternatives, pros/cons, decision factors)",
            priority=GeoPriority.MEDIUM,
            minutes=30,
            why="Commercial intent queries are often answered via comparisons and decision criteria.",
            how=(
                "Add `## Alternatives` or `## Comparison` and include: who it’s for, pricing range, key features, "
                "tradeoffs, and a short table if possible."
            ),
            examples="Headings: `## Who this is for` `## Pros and cons` `## Alternatives` `## Comparison table`",
        ))

    # 4) Structure for skimmability / retrieval
    if signals.h2_count == 0 and signals.h3_count == 0 and signals.word_count >= 400:
        steps.append(GeoNextStep(
            title="Add clear H2/H3 headings that match user questions",
            priority=GeoPriority.MEDIUM,
            minutes=20,
            why=(
                "Headings create extractable chunks and help retrieval/summarization. "
                "They also make it easier to cover subtopics without drifting."
            ),
            how=(
                "Add descriptive sections where they improve scanning. Use question headings only when questions "
                "match how the audience approaches this page; a homepage or landing page may use benefit-led headings."
            ),
            examples="Example headings: `## What is <TOPIC>?` `## When to use <TOPIC>` `## Steps` `## FAQ`",
        ))

    # 5) Evidence / citations
    if (page_type in (PageType.ARTICLE, PageType.HOW_TO, PageType.REFERENCE)
            and (signals.link_count == 0 or not signals.has_sources_section)
            and signals.word_count >= 300):
        steps.append(GeoNextStep(
            title="Add evidence: cite reputable sources and link out",
            priority=GeoPriority.MEDIUM,
            minutes=15,
            why=(
                "Important factual claims are easier for readers to verify when they point to primary or authoritative sources."
            ),
            how=(
                "Add 2–5 outbound links to authoritative sources for key claims/definitions. "
                "Prefer primary sources (standards, docs) or widely recognized publications."
            ),
            examples="Cite sources next to the relevant claims, or add a Sources section when that format suits the page.",
        ))

    # 5b) Add examples (quoteability + disambiguation)
    if page_type in (PageType.ARTICLE, PageType.HOW_TO, PageType.REFERENCE) and not signals.has_examples and signals.word_count >= 400:
        steps.append(GeoNextStep(
            title="Add 2–3 concrete examples (entities, numbers, scenarios)",
            priority=GeoPriority.MEDIUM if report.coverage.score < 80 else GeoPriority.LOW,
            minutes=15,
            why="Examples can reduce vagueness and make an explanation easier for readers to apply.",
            how="Add a short examples subsection under the most important headings. Include at least one numeric detail.",
        ))

    # 6) FAQ / intent coverage
    if page_type in (PageType.ARTICLE, PageType.HOW_TO, PageType.REFERENCE) and not signals.has_faq and signals.word_count >= 500:
        priority = GeoPriority.MEDIUM if report.coverage.score < 70 else GeoPriority.LOW
        steps.append(GeoNextStep(
            title="Add an FAQ that answers the top 5–8 questions",
            priority=priority,
            minutes=25,
            why=(
                "An FAQ can cover genuine follow-up questions compactly, but only when those questions do not fit "
                "more naturally in the main narrative."
            ),
            how=(
                "Consider an FAQ for recurring audience questions. Skip it when it would duplicate the article or "
                "turn natural prose into a search template."
            ),
        ))

    # 7) TL;DR / summary block
    if page_type in (PageType.ARTICLE, PageType.HOW_TO, PageType.REFERENCE) and not signals.has_tldr and signals.word_count >= 700:
        steps.append(GeoNextStep(
            title="Consider a short summary for readers who scan",
            priority=GeoPriority.LOW,
            minutes=10,
            why=(
                "A short summary can improve scanability on a long explanatory page. It is optional, not a universal format requirement."
            ),
            how="If it suits the voice, add two to four bullets with the main conclusions or actions. A descriptive standfirst can work just as well.",
        ))

    # Summary
    ccs = report.coverage.score
    if ccs >= 80:
        summary = "Strong topical focus for this target. Review the optional signals below in the context of the page's purpose."
    elif ccs >= 60:
        summary = "Decent topical focus. Prioritize reducing drift and strengthening weak sections."
    elif ccs >= 40:
        summary = "Partial topical focus. Check whether the target accurately describes the page before restructuring it."
    else:
        summary = "Low topical alignment. Confirm the target first; the page may be serving a different purpose."

    # Stable ordering: HIGH -> MEDIUM -> LOW, then shortest time first
    priority_order = {GeoPriority.HIGH: 0, GeoPriority.MEDIUM: 1, GeoPriority.LOW: 2}
    steps_sorted = sorted(steps, key=lambda s: (priority_order[s.priority], s.minutes))[:max_steps]

    return GeoNextStepsReport(
        summary=summary,
        steps=steps_sorted,
        signals=signals,
        intent=intent,
        page_type=page_type,
    )
