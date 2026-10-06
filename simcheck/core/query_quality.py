"""Target-query guardrails and page-derived query suggestions.

Similarity scores are only meaningful relative to a usable target.  This
module keeps obviously placeholder input from reaching the embedding pipeline
and gives the UI a lightweight way to help users replace vague targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_HTML_HEADING_RE = re.compile(r"<h[1-3]\b[^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\b(?:placeholder|lorem ipsum|dummy|sample query|test query|test topic)\b", re.IGNORECASE),
    re.compile(r"^(?:tbd|todo|n/?a|none|unknown|query|topic)$", re.IGNORECASE),
    re.compile(r"^needs?\s+(?:match|matched|matching)$", re.IGNORECASE),
)
_GENERIC_HEADINGS = {
    "home", "about", "contact", "menu", "navigation", "learn more", "read more",
    "faq", "frequently asked questions", "summary", "sources", "references",
}


@dataclass(frozen=True)
class QueryAssessment:
    """Whether a target can be scored, plus non-blocking quality guidance."""

    usable: bool
    error: str | None = None
    warning: str | None = None


def assess_target_query(query: str) -> QueryAssessment:
    """Reject empty/placeholder targets and flag unusually thin ones."""
    value = " ".join((query or "").split())
    if not value:
        return QueryAssessment(False, "Enter the question or topic this page should answer.")
    if any(pattern.search(value) for pattern in _PLACEHOLDER_PATTERNS):
        return QueryAssessment(
            False,
            "This looks like a placeholder, so SimCheck will not produce a content-fit score. "
            "Use a real searcher question or a specific topic phrase.",
        )

    words = _WORD_RE.findall(value)
    if not words:
        return QueryAssessment(False, "The target needs at least one word or named entity.")
    if len({word.lower() for word in words}) == 1 and len(words) > 2:
        return QueryAssessment(False, "The target repeats one term and is too ambiguous to score reliably.")
    if len(words) <= 2:
        return QueryAssessment(
            True,
            warning=(
                "Short target: results are directional. A searcher-shaped phrase or question "
                "usually gives a more dependable comparison."
            ),
        )
    return QueryAssessment(True)


def suggest_target_queries(document: str, limit: int = 3) -> list[str]:
    """Return editable topic candidates from the document's own headings.

    These describe what the page currently appears to cover; they are not
    claims about what users actually search for.
    """
    candidates = _HEADING_RE.findall(document or "")
    candidates += [_TAG_RE.sub(" ", h) for h in _HTML_HEADING_RE.findall(document or "")]

    suggestions: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        value = re.sub(r"[`*_\[\]]", "", raw)
        value = re.sub(r"\s+", " ", value).strip(" :-|#")
        key = value.casefold()
        if len(value) < 4 or key in _GENERIC_HEADINGS or key in seen:
            continue
        if len(_WORD_RE.findall(value)) > 14:
            continue
        seen.add(key)
        suggestions.append(value)
        if len(suggestions) >= limit:
            break
    return suggestions
