"""
Executive summary for the Report tab and the shareable client report.

Turns the analysis (page quality, AI access, content patterns, optional probes)
into a headline sentence, two short summary paragraphs, and a ranked
"fix first" list. Deterministic templates, no LLM: every sentence traces to
a measured signal, and the same inputs always give the same report.

Copy rules: plain declarative sentences, no hype words, no em-dash
asides, numbers stated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from simcheck.core.geo import GeoNextStepsReport, GeoPriority
from simcheck.core.readiness import ReadinessScore
from simcheck.quality.ai_access import AIAccessReport
from simcheck.quality.page_quality import PageQualityRating
from simcheck.quality.probe import ProbeReport
from simcheck.quality.snapshot import PageSnapshot


FIX_LIMIT = 3

# Fix priority tiers (lower sorts first)
_TIER_ACCESS = 0
_TIER_GATE = 1
_TIER_GEO_HIGH = 2
_TIER_QUALITY = 3
_TIER_GEO_OTHER = 4


@dataclass(frozen=True)
class Fix:
    """One recommended change."""
    title: str
    detail: str
    minutes: Optional[int]
    source: str  # "access" | "quality" | "content"
    tier: int = _TIER_GEO_OTHER


@dataclass(frozen=True)
class PageReport:
    """Everything the Report tab and shareable report render."""
    headline: str
    summary: tuple          # paragraphs
    fixes: tuple            # Fix, ranked
    quality_line: str       # one-line caption under the PQ number
    visibility_line: str    # caption under the citations / access number
    simscore_line: str      # legacy field name; caption under Content patterns


# =============================================================================
# Phrases
# =============================================================================

def _month_year(iso: Optional[str]) -> Optional[str]:
    """'2025-05-12T13:29:32Z' -> 'May 2025'; None if unparseable."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso[:10]).strftime("%B %Y")
    except ValueError:
        return None


def _join(items: list) -> str:
    """Oxford-comma list: a, b, and c."""
    items = [i for i in items if i]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _quality_clause(rating: Optional[PageQualityRating]) -> Optional[str]:
    if rating is None:
        return None
    if not rating.rated:
        return "SimCheck can't see this page's content"
    return f"Google would rate this page {rating.band}"


def _visibility_clause(access: Optional[AIAccessReport], probes: Optional[ProbeReport],
                       readiness: Optional[ReadinessScore]) -> Optional[str]:
    """Second half of the headline, strongest available evidence first."""
    if access is not None:
        if access.noindex:
            return "It is set to noindex, so search and AI engines won't show it"
        if access.search_bots_blocked:
            return "It blocks the crawlers AI search engines use"
        if access.client_rendered_suspect:
            return "Neither can AI crawlers that don't run JavaScript"
    if probes is not None and probes.completed:
        n, k = len(probes.completed), probes.cited_count
        if k == 0:
            return "AI search engines aren't citing it"
        if k == n:
            return "AI search engines cite it"
        return f"AI search engines cite it in {k} of {n} answers"
    if access is not None:
        return "AI search crawlers can reach it, but citation has not been tested"
    return None


def build_headline(rating, access, probes, readiness) -> str:
    """Two-clause finding sentence for the top of the report."""
    first = _quality_clause(rating)
    second = _visibility_clause(access, probes, readiness)
    if first and second:
        return f"{first}. {second}."
    return f"{first or second or 'Analysis complete'}."


def _quality_paragraph(snapshot: PageSnapshot, rating: PageQualityRating) -> str:
    if not rating.rated:
        return rating.reasons[0]
    level = rating.level or 0
    bar = "clears" if level >= 3 else ("meets" if level >= 2 else "falls short of")
    topic = (f"a {rating.ymyl_topic} topic where accuracy matters"
             if rating.ymyl == "clearly" and rating.ymyl_topic not in (None, "none")
             else "this kind of page")
    facts = []
    facts.append("names its author" if snapshot.author_name else "doesn't name an author")
    schema = [t for t in snapshot.schema_types if t not in ("WebSite", "SearchAction", "ImageObject", "BreadcrumbList")]
    if schema:
        facts.append(f"uses {schema[0]} schema")
    if snapshot.external_link_count:
        facts.append(f"links {snapshot.external_link_count} outside sources")
    updated = _month_year(snapshot.modified or snapshot.published)
    if updated:
        facts.append(f"was updated in {updated}" if snapshot.modified else f"was published in {updated}")
    text = f"The page {bar} Google's bar for {topic}. It {_join(facts)}."
    if rating.gates:
        text += " " + rating.gates[0]
    return text


def _visibility_paragraph(access, probes, readiness) -> Optional[str]:
    parts = []
    if access is not None:
        if access.search_bots_blocked:
            parts.append(f"robots.txt blocks {_join(list(access.search_bots_blocked))}, "
                         "so those engines can't use it in answers.")
        elif not access.noindex:
            parts.append("Every AI search crawler can reach it.")
    if probes is not None and probes.completed:
        n, k = len(probes.completed), probes.cited_count
        competitors = [h for h, _ in probes.top_competitors(3)]
        if k == 0:
            line = f"It wasn't cited in any of the {n} AI answers we tested"
            if competitors:
                line += f"; {_join(competitors)} were"
            parts.append(line + ".")
        else:
            parts.append(f"It was cited in {k} of {n} AI answers we tested.")
    return " ".join(parts) or None


# =============================================================================
# Fixes
# =============================================================================

def _access_fixes(access: AIAccessReport) -> list:
    fixes = []
    if access.noindex:
        fixes.append(Fix("Remove noindex", "Nothing else matters until search engines may index the page.",
                         5, "access", _TIER_ACCESS))
    if access.search_bots_blocked:
        fixes.append(Fix("Allow AI search crawlers in robots.txt",
                         f"Unblock {_join(list(access.search_bots_blocked))}.", 10, "access", _TIER_ACCESS))
    if access.client_rendered_suspect:
        fixes.append(Fix("Serve the content in the HTML",
                         "Use server-side rendering or prerendering so crawlers see the text.",
                         None, "access", _TIER_ACCESS))
    if access.nosnippet:
        fixes.append(Fix("Allow snippets", "Remove nosnippet / max-snippet:0 so engines can quote the page.",
                         5, "access", _TIER_ACCESS))
    return fixes


def _quality_fixes(snapshot: PageSnapshot, rating: PageQualityRating) -> list:
    fixes = []
    if not rating.rated:
        return fixes
    for gate in rating.gates:
        if "author" in gate:
            fixes.append(Fix("Name the author and who runs the site",
                             "Add a byline with credentials and link About and Contact pages.",
                             20, "quality", _TIER_GATE))
        elif "Ads" in gate:
            fixes.append(Fix("Cut ads and affiliate links around the main content",
                             "Raters cap pages at Low when monetization dominates.", 30, "quality", _TIER_GATE))
        elif "mass-produced" in gate:
            fixes.append(Fix("Rewrite with original substance",
                             "Add first-hand testing, data, or expert input the template lacks.",
                             None, "quality", _TIER_GATE))
    eeat = rating.eeat
    if eeat and int(eeat.get("experience", 2) + 0.5) <= 2 and rating.ymyl == "clearly":
        fixes.append(Fix("Add a first-hand perspective",
                         "A clinician's or practitioner's note lifts Experience, the weakest trust signal.",
                         30, "quality", _TIER_QUALITY))
    if not snapshot.author_name and not any(f.tier == _TIER_GATE for f in fixes):
        fixes.append(Fix("Add a named author", "Raters look for who wrote the content and why they're qualified.",
                         10, "quality", _TIER_QUALITY))
    return fixes


def _content_fixes(geo: GeoNextStepsReport) -> list:
    out = []
    for step in geo.steps:
        tier = _TIER_GEO_HIGH if step.priority == GeoPriority.HIGH else _TIER_GEO_OTHER
        detail = "Editorial option: " + step.how.split("\n")[0].strip()
        out.append(Fix(step.title, detail, step.minutes, "content", tier))
    return out


def rank_fixes(fixes: list, limit: int = FIX_LIMIT) -> tuple:
    """Order by tier, then quickest first, dropping duplicate titles."""
    seen, ranked = set(), []
    for f in sorted(fixes, key=lambda f: (f.tier, f.minutes if f.minutes is not None else 999)):
        if f.title not in seen:
            seen.add(f.title)
            ranked.append(f)
    return tuple(ranked[:limit])


# =============================================================================
# Public API
# =============================================================================

def build_report(
    snapshot: Optional[PageSnapshot] = None,
    rating: Optional[PageQualityRating] = None,
    access: Optional[AIAccessReport] = None,
    readiness: Optional[ReadinessScore] = None,
    geo: Optional[GeoNextStepsReport] = None,
    probes: Optional[ProbeReport] = None,
) -> PageReport:
    """
    Build the executive summary from whatever parts of the analysis ran.

    Every input is optional so the report degrades gracefully: no keys
    (no rating), no query (no content-pattern score), probes not run, or pasted text
    (no snapshot/access).

    Returns:
        PageReport
    """
    summary = []
    if snapshot is not None and rating is not None:
        summary.append(_quality_paragraph(snapshot, rating))
    vis = _visibility_paragraph(access, probes, readiness)
    if vis:
        summary.append(vis)

    fixes = []
    if access is not None:
        fixes += _access_fixes(access)
    if snapshot is not None and rating is not None:
        fixes += _quality_fixes(snapshot, rating)
    if geo is not None:
        fixes += _content_fixes(geo)

    quality_line = ""
    if rating is not None:
        if not rating.rated:
            quality_line = "Not rated: no server-rendered content."
        elif rating.gates:
            quality_line = rating.gates[0]
        else:
            ymyl = f"YMYL {rating.ymyl_topic} page" if rating.ymyl == "clearly" and rating.ymyl_topic != "none" else "Page"
            quality_line = f"{ymyl}, trust rated {['Lowest', 'Low', 'Medium', 'High', 'Highest'][int(rating.eeat['trust'] + 0.5)]}."

    if probes is not None and probes.completed:
        competitors = [h for h, _ in probes.top_competitors(3)]
        visibility_line = (f"Cited instead: {_join(competitors)}." if probes.cited_count == 0 and competitors
                           else f"Cited in {probes.cited_count} of {len(probes.completed)} answers.")
    elif access is not None:
        visibility_line = ("AI search crawlers blocked." if access.search_bots_blocked
                           else "All AI search crawlers allowed. Run probes to check citations.")
    else:
        visibility_line = ""

    simscore_line = ""
    if readiness is not None:
        simscore_line = (
            f"Experimental heuristic. Target coverage: {round(readiness.components['coverage'])}/100. "
            "This does not predict citations."
        )

    return PageReport(
        headline=build_headline(rating, access, probes, readiness),
        summary=tuple(summary),
        fixes=rank_fixes(fixes),
        quality_line=quality_line,
        visibility_line=visibility_line,
        simscore_line=simscore_line,
    )


def site_headline(summary: dict) -> str:
    """
    Finding sentence for the Site Audit tab.

    Args:
        summary: SiteAudit.summary()

    Returns:
        One or two sentences
    """
    n = summary.get("rated", 0)
    if n == 0:
        return "No sampled page could be rated."
    high, low = summary["high_or_better"], summary["low_or_worse"]
    if low:
        return f"{low} of {n} sampled pages rate Low or worse. Start there."
    if high == n:
        return ("Every sampled page rates High or better. Quality is uniform, "
                "so it isn't what separates these pages in AI answers.")
    return f"{high} of {n} sampled pages rate High or better. The rest sit at Medium."
