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

from simcheck.core.diagnostics import DiagnosticReport, ChunkDiagnostic
from simcheck.core.models import SIMILARITY_THRESHOLDS


class GeoIntent(Enum):
    AUTO = "auto"
    INFORMATIONAL = "informational"
    HOW_TO = "how_to"
    COMMERCIAL = "commercial"


class GeoPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ContentSignals:
    """
    Simple structure/evidence signals that correlate with "AI answerability".

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
_HTML_H2_RE = re.compile(r"<h2\\b", re.IGNORECASE)
_HTML_H3_RE = re.compile(r"<h3\\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"\b(is|are|refers to|means)\b", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"\b(for example|e\.g\.|example:)\b", re.IGNORECASE)
_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compare|comparison|alternatives?)\b", re.IGNORECASE)
_FRESHNESS_RE = re.compile(r"\b(updated|last updated|as of|new in)\b|\b20(1\\d|2\\d)\\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\\d")


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
        normalized_heading = re.sub(r"^\\s{0,3}#+\\s*", "", line).strip()
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
    max_steps: int = 7,
) -> GeoNextStepsReport:
    """
    Generate GEO-oriented next steps based on the diagnostic report + raw text.

    This complements (not replaces) CCS recommendations, focusing on actions a
    content editor can take quickly: front-load answers, reduce drift, improve
    structure, and add evidence.
    """
    intent = infer_intent(report.query) if intent_override == GeoIntent.AUTO else intent_override
    signals = extract_content_signals(document, report.query)

    total_chunks = report.summary.total_chunks
    if total_chunks <= 0:
        return GeoNextStepsReport(
            summary="No chunks found; add content and re-run analysis.",
            steps=[],
            signals=signals,
            intent=intent,
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

    weak_threshold = SIMILARITY_THRESHOLDS["weak"]
    moderate_threshold = SIMILARITY_THRESHOLDS["moderate"]
    weak_chunks = [c for c in report.chunks if weak_threshold <= c.similarity < moderate_threshold]

    steps: List[GeoNextStep] = []

    # 1) Front-load the answer (GEO critical)
    front_load_needed = (
        signals.intro_query_term_coverage < 0.5
        or best_pos > 0.4
        or (intro_avg + 0.05) < overall_avg
    )
    if front_load_needed:
        target = [c for c in doc_chunks[: min(3, len(doc_chunks))]]
        steps.append(GeoNextStep(
            title="Front-load a direct answer + definition (first ~150 words)",
            priority=GeoPriority.HIGH,
            minutes=15,
            why=(
                "AI systems heavily weight early sections when summarizing or selecting snippets. "
                "If the intro doesn’t clearly name and define the topic, relevance is harder to infer."
            ),
            how=(
                "Add a 2–4 sentence lead that (1) defines the topic, (2) states who it’s for / when it applies, "
                "and (3) previews the main subtopics you cover. Reuse exact terms from your target topic."
            ),
            examples=(
                "Template:\n"
                "“<TOPIC> is <definition>. In this guide you’ll learn <3 key subtopics>. "
                "If you’re <audience/intent>, start with <first recommended action>.”"
            ),
            target_chunks=target,
        ))

    # 1b) Add a crisp definition near the top (especially informational queries)
    if intent in (GeoIntent.INFORMATIONAL, GeoIntent.HOW_TO) and not signals.has_definition_near_top:
        steps.append(GeoNextStep(
            title="Add a crisp definition near the top (1–2 sentences)",
            priority=GeoPriority.HIGH if report.coverage.score < 70 else GeoPriority.MEDIUM,
            minutes=10,
            why="Clear definitions reduce ambiguity and increase quoteability in AI summaries.",
            how=(
                "Add a sentence that directly defines the target topic using the exact name, then follow with "
                "a second sentence that clarifies scope (what it includes/excludes)."
            ),
        ))

    # 2) Remove or rewrite off-topic content (CCS + GEO)
    if off_topic_percent >= 0.10:
        priority = GeoPriority.HIGH if off_topic_percent >= 0.15 else GeoPriority.MEDIUM
        steps.append(GeoNextStep(
            title="Rewrite or cut off-topic sections (reduce topical drift)",
            priority=priority,
            minutes=20,
            why=(
                "Off-topic sections dilute topical focus and can prevent AI systems from confidently "
                "treating the page as an authoritative answer for the target topic."
            ),
            how=(
                "For each off-topic chunk: either (a) connect it back to the target topic with a clear bridge "
                "sentence and relevant examples, or (b) remove/condense it. Aim to get these chunks above 0.45."
            ),
            target_chunks=off_topic[:5],
        ))

    # 3) Strengthen weak chunks into "moderate"
    if weak_chunks:
        steps.append(GeoNextStep(
            title="Strengthen weak sections with concrete specifics (raise to ≥0.65)",
            priority=GeoPriority.MEDIUM,
            minutes=25,
            why=(
                "Weak chunks signal partial relevance. Adding specific entities, constraints, examples, "
                "and explicit mentions of the target concept typically improves alignment."
            ),
            how=(
                "Add: key terms, named tools/entities, numbers, and a short example. "
                "Replace vague language (“this”, “it”, “some”) with explicit references to the topic."
            ),
            target_chunks=weak_chunks[:5],
        ))

    # 3b) Intent-specific “answerability” structure
    if intent == GeoIntent.HOW_TO and not signals.has_steps:
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

    if intent == GeoIntent.COMMERCIAL and not signals.has_comparison_language and signals.word_count >= 400:
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
                "Add 4–8 `##` sections covering the main subtopics. For each, add a one-paragraph direct answer "
                "and (optionally) bullets or a table."
            ),
            examples="Example headings: `## What is <TOPIC>?` `## When to use <TOPIC>` `## Steps` `## FAQ`",
        ))

    # 5) Evidence / citations
    if (signals.link_count == 0 or not signals.has_sources_section) and signals.word_count >= 300:
        steps.append(GeoNextStep(
            title="Add evidence: cite reputable sources and link out",
            priority=GeoPriority.MEDIUM,
            minutes=15,
            why=(
                "LLMs are more likely to trust and cite content that anchors claims in reputable sources. "
                "Outbound links also clarify definitions and entities."
            ),
            how=(
                "Add 2–5 outbound links to authoritative sources for key claims/definitions. "
                "Prefer primary sources (standards, docs) or widely recognized publications."
            ),
            examples="Add a short `## Sources` section with bullet links.",
        ))

    # 5b) Add examples (quoteability + disambiguation)
    if not signals.has_examples and signals.word_count >= 400:
        steps.append(GeoNextStep(
            title="Add 2–3 concrete examples (entities, numbers, scenarios)",
            priority=GeoPriority.MEDIUM if report.coverage.score < 80 else GeoPriority.LOW,
            minutes=15,
            why="Examples reduce vagueness and help AI systems extract specific, reusable claims.",
            how="Add a short examples subsection under the most important headings. Include at least one numeric detail.",
        ))

    # 6) FAQ / intent coverage
    if not signals.has_faq and signals.word_count >= 500:
        priority = GeoPriority.MEDIUM if report.coverage.score < 70 else GeoPriority.LOW
        steps.append(GeoNextStep(
            title="Add an FAQ that answers the top 5–8 questions",
            priority=priority,
            minutes=25,
            why=(
                "FAQs expand intent coverage and create highly quotable question/answer pairs. "
                "This tends to help AI summaries and conversational search."
            ),
            how=(
                "Add a `## FAQ` section. Use question headings (e.g., `### ...?`) and answer each in 2–4 sentences. "
                "Include the target terms in the question and first sentence of each answer."
            ),
        ))

    # 7) TL;DR / summary block
    if not signals.has_tldr and signals.word_count >= 700:
        steps.append(GeoNextStep(
            title="Add a TL;DR box (2–4 bullets) near the top",
            priority=GeoPriority.LOW,
            minutes=10,
            why=(
                "A short summary improves scanability and gives AI systems a concise set of claims to reuse."
            ),
            how="Add `TL;DR:` followed by 2–4 bullets stating the main conclusions and recommended actions.",
        ))

    # Summary
    ccs = report.coverage.score
    if ccs >= 80:
        summary = "Strong topical focus. Prioritize front-loading + evidence to increase citation/summary likelihood."
    elif ccs >= 60:
        summary = "Decent topical focus. Prioritize reducing drift and strengthening weak sections."
    elif ccs >= 40:
        summary = "Weak topical focus. Prioritize a clearer intro + restructuring around the target topic."
    else:
        summary = "Low topical focus. The content likely doesn’t answer the target topic directly yet."

    # Stable ordering: HIGH -> MEDIUM -> LOW, then shortest time first
    priority_order = {GeoPriority.HIGH: 0, GeoPriority.MEDIUM: 1, GeoPriority.LOW: 2}
    steps_sorted = sorted(steps, key=lambda s: (priority_order[s.priority], s.minutes))[:max_steps]

    return GeoNextStepsReport(
        summary=summary,
        steps=steps_sorted,
        signals=signals,
        intent=intent,
    )
