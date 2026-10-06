"""
Streamlit views for SimCheck v2: URL bar, Report, Page Quality,
LLM Visibility, and Site Audit tabs.

Thin layer: all analysis lives in simcheck.quality / simcheck.core.

Two Streamlit rules matter here (see CLAUDE.md):
- HTML passed to st.markdown must not contain newlines (they end the HTML
  block mid-element), so every block goes through _html(), which collapses
  whitespace.
- Every page-derived string is escaped with esc(): titles, URLs, and
  cited hosts come from third-party sites and st.markdown renders raw HTML.
"""

from __future__ import annotations

from html import escape
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from simcheck.config import ConfigError, load_api_keys
from simcheck.core.geo import GeoIntent, PageType, generate_geo_next_steps
from simcheck.core.query_quality import assess_target_query
from simcheck.core.readiness import compute_readiness_score
from simcheck.quality.ai_access import AI_BOTS, SEARCH_CRITICAL_BOTS, check_ai_access
from simcheck.quality.classifier import ClassifierError, build_state, make_classifier, make_explainer
from simcheck.quality.export import analysis_json, audit_csv, band_color, share_report_html
from simcheck.quality.llm_client import LLMError, OpenRouterClient
from simcheck.quality.page_quality import rate_page
from simcheck.quality.probe import (
    DEFAULT_ENGINE,
    MAX_PROBE_QUERIES,
    PROBE_ENGINES,
    ProbeError,
    default_queries,
    estimate_cost,
    run_probes,
)
from simcheck.quality.report import build_report, site_headline
from simcheck.quality.rubric import PQ_LEVELS, PQ_SLIDER, QUESTIONS_BY_ID
from simcheck.quality.site import DEFAULT_SAMPLE, MAX_SAMPLE, SiteAuditError, audit_site
from simcheck.quality.snapshot import FetchBlockedError, SnapshotError, parse_snapshot, snapshot_url
from ui.access_views import guard, max_site_sample


INK = "#15181D"
MUTED = "#5B6270"
BODY = "#2A2F37"
RULE = "#E3E6EA"
BAD = "#B4541A"

# Questions Claude explains on "Explain this rating"
EXPLAIN_QUESTIONS = ("page_quality", "trust", "expertise", "experience", "mc_quality")

QUALITY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');
.sc { font-family: 'Geist', system-ui, sans-serif; color: #15181D; }
.sc-meta { font-size: 13px; color: #5B6270; margin: 8px 0 0; }
.sc-h1 { font-size: 42px; line-height: 1.12; font-weight: 600; letter-spacing: -0.025em; margin: 20px 0 0; max-width: 880px; }
.sc-h1-sm { font-size: 32px; line-height: 1.18; font-weight: 600; letter-spacing: -0.02em; margin: 16px 0 0; max-width: 860px; }
.sc-figs { display: grid; border-top: 1px solid #15181D; margin-top: 48px; }
.sc-fig { padding: 22px 28px 0 28px; border-left: 1px solid #E3E6EA; }
.sc-fig:first-child { padding-left: 0; border-left: 0; }
.sc-label { font-size: 13px; color: #5B6270; }
.sc-num { font-size: 52px; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1; margin-top: 8px; }
.sc-num-sm { font-size: 32px; font-weight: 600; letter-spacing: -0.02em; margin-top: 6px; }
.sc-sub { font-family: 'Geist Mono', monospace; font-size: 17px; color: #5B6270; margin-left: 12px; font-weight: 400; letter-spacing: 0; }
.sc-cap { font-size: 14px; line-height: 1.5; color: #3A404A; margin: 8px 0 0; }
.sc-sec { display: grid; grid-template-columns: 220px 1fr; column-gap: 32px; margin-top: 64px; }
.sc-sec h2 { font-size: 15px; font-weight: 600; margin: 0; padding: 0; }
.sc-body p { font-size: 18px; line-height: 1.6; color: #2A2F37; margin: 0 0 16px; }
.sc-fixes { list-style: none; margin: 0; padding: 0; border-top: 1px solid #E3E6EA; }
.sc-fixes li { display: grid; grid-template-columns: 40px 1fr 80px; gap: 16px; padding: 18px 0; border-bottom: 1px solid #E3E6EA; margin: 0; }
.sc-idx { font-family: 'Geist Mono', monospace; color: #5B6270; }
.sc-fix-t { font-weight: 500; font-size: 16px; }
.sc-fix-d { font-size: 14px; color: #5B6270; margin-top: 4px; }
.sc-min { font-size: 14px; color: #5B6270; text-align: right; }
.sc-dl { display: grid; grid-template-columns: 200px 1fr; border-top: 1px solid #E3E6EA; margin: 0; font-size: 15px; }
.sc-dl dt { padding: 13px 0; color: #5B6270; border-bottom: 1px solid #E3E6EA; }
.sc-dl dd { padding: 13px 0; margin: 0; border-bottom: 1px solid #E3E6EA; }
.sc-mono { font-family: 'Geist Mono', monospace; font-size: 13px; }
.sc-scale { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 4px; margin-top: 32px; }
.sc-scale div { height: 8px; background: #E8EAED; }
.sc-scale-l { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 4px; margin-top: 10px; font-size: 12px; color: #5B6270; }
.sc-bar { display: grid; grid-template-columns: 160px 1fr 72px; gap: 20px; align-items: center; margin-bottom: 20px; }
.sc-track { height: 6px; background: #E8EAED; }
.sc-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.sc-table th { text-align: left; font-weight: 400; font-size: 13px; color: #5B6270; padding: 10px 8px 10px 0; border-bottom: 1px solid #15181D; }
.sc-table td { padding: 14px 8px 14px 0; border-bottom: 1px solid #E3E6EA; vertical-align: top; }
.sc-note { font-size: 13px; color: #5B6270; margin-top: 12px; }
.sc-empty { font-size: 15px; color: #5B6270; padding: 48px 0; }
@media (max-width: 760px) {
  .sc-h1 { font-size: 30px; } .sc-figs { grid-template-columns: 1fr !important; }
  .sc-fig { padding-left: 0; border-left: 0; padding-bottom: 16px; }
  .sc-sec { grid-template-columns: 1fr; row-gap: 12px; }
}
</style>
"""


def esc(value) -> str:
    """Escape any value for HTML."""
    return escape("" if value is None else str(value))


def _html(markup: str) -> None:
    """Render an HTML block (whitespace collapsed; see module docstring)."""
    st.markdown(" ".join(markup.split()), unsafe_allow_html=True)


def inject_css() -> None:
    """Add the v2 stylesheet (call once per run)."""
    _html(QUALITY_CSS)


def _keys():
    """API keys, or None (with an error shown) if the key file is unsafe."""
    try:
        return load_api_keys()
    except ConfigError as e:
        st.error(str(e))
        return None


def _state():
    return st.session_state


# =============================================================================
# Analysis
# =============================================================================

def init_quality_state() -> None:
    """Session-state defaults for the v2 views."""
    defaults = {
        "qa_url": "",
        "qa_query": "",
        "qa_paste_html": "",
        "analysis": None,       # dict: url, query, snapshot, access, rating, blocked, error
        "probes": None,
        "probe_queries": "",
        "explanation": None,
        "site_audit": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _analyze(url: str, query: str, pasted_html: str, run_content_match: Callable) -> None:
    """Run snapshot → access → rating → Content Match; store results in session state."""
    s = _state()
    s.probes, s.explanation = None, None
    analysis = {"url": url, "query": query, "snapshot": None, "access": None,
                "rating": None, "blocked": None, "error": None, "rating_error": None}
    try:
        snapshot = parse_snapshot(url, pasted_html) if pasted_html.strip() else snapshot_url(url)
    except FetchBlockedError as e:
        analysis["blocked"] = str(e)
        s.analysis = analysis
        return
    except SnapshotError as e:
        analysis["error"] = str(e)
        s.analysis = analysis
        return
    analysis["snapshot"] = snapshot

    try:
        analysis["access"] = check_ai_access(snapshot, query)
    except SnapshotError as e:
        analysis["error"] = f"AI access check failed: {e}"

    keys = _keys()
    classifier = make_classifier(keys) if keys else None
    if classifier is not None:
        try:
            analysis["rating"] = rate_page(snapshot, classifier, analysis["access"], query or None)
        except ClassifierError as e:
            analysis["rating_error"] = str(e)
    s.analysis = analysis
    s.probe_queries = "\n".join(default_queries(query, snapshot.title))

    # Feed Content Match: these widget keys render later in this run, so
    # setting them here is allowed.
    if snapshot.main_markdown:
        s.query_input = query if query.strip() else ""
        s.document_input = snapshot.main_markdown
        s.strategy_select = "markdown"
        s.fetched_url = url
        if query.strip():
            run_content_match(query, snapshot.main_markdown, "markdown")
        else:
            s.is_indexed = False
            s.comparison_result = None
            s.diagnostic_report = None
            s.recommendation_report = None
            s.last_analyzed_query = ""
            s.last_analyzed_document = ""


def render_url_bar(run_content_match: Callable) -> None:
    """Top input row: URL, target query, Analyze. Handles the blocked-page paste fallback."""
    with st.container():
        c1, c2, c3 = st.columns([6, 4, 1.3], vertical_alignment="bottom")
        with c1:
            st.text_input("Page URL", key="qa_url", placeholder="https://example.com/page")
        with c2:
            st.text_input("Target query", key="qa_query", placeholder="what a searcher would ask",
                          help="Optional. Adds Needs Met and Content Match. Placeholder targets are not scored.")
        with c3:
            go = st.button("Analyze", type="primary", use_container_width=True)

    a = _state().analysis
    if a and a.get("blocked"):
        with st.expander("This site blocks automated fetchers. Paste the page HTML to rate it anyway.", expanded=True):
            st.text_area("Page HTML (View source, then copy all)", key="qa_paste_html", height=140)

    if go:
        url = _state().qa_url.strip()
        if not url:
            st.warning("Enter a page URL.")
            return
        if "//" not in url:
            url = "https://" + url
        query = _state().qa_query.strip()
        assessment = assess_target_query(query) if query else None
        if assessment and not assessment.usable:
            st.warning(assessment.error + " The page-quality and access checks will still run without it.")
            query = ""
        elif assessment and assessment.warning:
            st.warning(assessment.warning)
        if not guard("analyze"):
            return
        with st.spinner("Fetching and rating the page..."):
            _analyze(url, query, _state().qa_paste_html, run_content_match)


def _readiness_and_geo():
    """Experimental content-pattern score + editorial options, if available."""
    s = _state()
    report = s.get("diagnostic_report")
    document = s.get("last_analyzed_document") or ""
    if not report or not document.strip():
        return None, None
    analysis = s.get("analysis") or {}
    rating = analysis.get("rating")
    snapshot = analysis.get("snapshot")
    geo = generate_geo_next_steps(
        report,
        document,
        intent_override=GeoIntent(s.get("geo_intent", "auto")),
        page_type_override=PageType(s.get("page_type", "auto")),
        page_url=snapshot.final_url if snapshot else s.get("fetched_url", ""),
        classified_purpose=rating.purpose if rating else None,
    )
    return compute_readiness_score(report, geo.signals, geo.intent, geo.page_type), geo


def _no_analysis() -> bool:
    a = _state().analysis
    if a is None:
        _html('<div class="sc sc-empty">Enter a URL above and press Analyze.</div>')
        return True
    if a.get("blocked"):
        st.warning(a["blocked"])
        return True
    if a.get("error") and a.get("snapshot") is None:
        st.error(a["error"])
        return True
    return False


# =============================================================================
# Report tab
# =============================================================================

def _figure(label: str, number: str, color: str, sub: str = "", caption: str = "", small: bool = False) -> str:
    sub_html = f'<span class="sc-sub">{esc(sub)}</span>' if sub else ""
    cls = "sc-num-sm" if small else "sc-num"
    return (f'<div class="sc-fig"><div class="sc-label">{esc(label)}</div>'
            f'<div class="{cls}" style="color:{color}">{esc(number)}{sub_html}</div>'
            f'<p class="sc-cap">{esc(caption)}</p></div>')


def render_report_tab() -> None:
    """Hero: headline finding, three figures, summary, fix-first list, exports."""
    if _no_analysis():
        return
    a, s = _state().analysis, _state()
    snapshot, access, rating = a["snapshot"], a["access"], a["rating"]
    readiness, geo = _readiness_and_geo()
    report = build_report(snapshot, rating, access, readiness, geo, s.probes)

    host = snapshot.host.removeprefix("www.")
    meta = f"{esc(host)} · Google QRG, September 2025 edition"
    left, right = st.columns([5, 2], vertical_alignment="bottom")
    with left:
        _html(f'<div class="sc"><p class="sc-meta">{meta}</p></div>')
    with right:
        e1, e2 = st.columns(2)
        e1.download_button("Share report", share_report_html(a["url"], report, rating, s.probes),
                           file_name=f"{host}-report.html", mime="text/html", use_container_width=True)
        e2.download_button("Export JSON", analysis_json(a["url"], a["query"], report=report, rating=rating,
                                                        access=access, readiness=readiness, probes=s.probes),
                           file_name=f"{host}-analysis.json", mime="application/json", use_container_width=True)

    figs = []
    if rating is not None:
        if rating.rated:
            figs.append(_figure("Page Quality", rating.band, band_color(rating.band), f"{rating.pq_score_rounded}/100",
                                report.quality_line))
        else:
            figs.append(_figure("Page Quality", "Unrated", MUTED, caption=report.quality_line))
    else:
        figs.append(_figure("Page Quality", "Not run", MUTED,
                            caption=a.get("rating_error") or "Add a TypeSafe key to rate page quality."))
    if s.probes is not None and s.probes.completed:
        k, n = s.probes.cited_count, len(s.probes.completed)
        figs.append(_figure("AI citations", f"{k} of {n}", BAD if k == 0 else INK, "answers", report.visibility_line))
    elif access is not None:
        blocked = bool(access.search_bots_blocked) or access.noindex
        figs.append(_figure("AI access", "Blocked" if blocked else "Open", BAD if blocked else INK,
                            caption=report.visibility_line))
    if readiness is not None:
        figs.append(_figure("Content patterns", str(round(readiness.score)), INK,
                            readiness.interpretation, report.simscore_line))
    else:
        figs.append(_figure("Content patterns", "–", MUTED,
                            caption="Add a usable target query to compare content patterns."))

    summary = "".join(f"<p>{esc(p)}</p>" for p in report.summary)
    fixes = "".join(
        f'<li><span class="sc-idx">{i:02d}</span><div><div class="sc-fix-t">{esc(f.title)}</div>'
        f'<div class="sc-fix-d">{esc(f.detail)}</div></div>'
        f'<span class="sc-min">{f"{f.minutes} min" if f.minutes else ""}</span></li>'
        for i, f in enumerate(report.fixes, 1))
    _html(
        f'<div class="sc"><h1 class="sc-h1">{esc(report.headline)}</h1>'
        f'<div class="sc-figs" style="grid-template-columns:repeat({len(figs)},minmax(0,1fr))">{"".join(figs)}</div>'
        + (f'<section class="sc-sec"><h2>Summary</h2><div class="sc-body">{summary}</div></section>' if summary else "")
        + (f'<section class="sc-sec"><h2>Editorial options</h2><ol class="sc-fixes">{fixes}</ol></section>' if fixes else "")
        + '</div>'
    )


# =============================================================================
# Page Quality tab
# =============================================================================

def _level_name(v: float) -> str:
    return PQ_LEVELS[int(min(max(v, 0), 4) + 0.5)]


def render_page_quality_tab() -> None:
    """Band on the 9-point scale, purpose/YMYL/Needs Met, E-E-A-T, rater-visible signals, explanation."""
    if _no_analysis():
        return
    a, s = _state().analysis, _state()
    snapshot, rating = a["snapshot"], a["rating"]
    if rating is None:
        st.info(a.get("rating_error") or "Add TYPESAFE_API_KEY to ~/.config/simcheck/.env to rate page quality.")
        return
    if not rating.rated:
        _html(f'<div class="sc"><div class="sc-num" style="color:{MUTED}">Unrated</div>'
              f'<p class="sc-cap" style="max-width:640px">{esc(rating.reasons[0])}</p></div>')
        return

    left, right = st.columns([5, 1.4], vertical_alignment="bottom")
    with left:
        _html(f'<div class="sc"><div class="sc-label">Page Quality, Google Search Quality Rater Guidelines</div>'
              f'<div class="sc-num" style="font-size:60px;color:{band_color(rating.band)}">{esc(rating.band)}'
              f'<span class="sc-sub">{rating.pq_score_rounded}/100</span></div></div>')
    keys = _keys()
    explainer = make_explainer(keys) if keys else None
    with right:
        if st.button("Explain this rating", disabled=explainer is None, use_container_width=True,
                     help=None if explainer else "Needs OPENROUTER_API_KEY") and guard("explain"):
            with st.spinner("Asking Claude for evidence..."):
                try:
                    questions = [QUESTIONS_BY_ID[q] for q in EXPLAIN_QUESTIONS]
                    s.explanation = explainer.classify(build_state(snapshot, a["access"], a["query"]), questions)
                except ClassifierError as e:
                    st.error(f"Explanation failed: {e}")

    idx = PQ_SLIDER.index(rating.band)
    cells = "".join(f'<div style="background:{band_color(rating.band) if i == idx else "#E8EAED"}"></div>'
                    for i in range(9))
    labels = "".join(
        f'<span style="{"color:" + band_color(rating.band) + ";font-weight:500" if i == idx else ""}">'
        f'{esc(name) if (i % 2 == 0 or i == idx) else ""}</span>' for i, name in enumerate(PQ_SLIDER))
    facts = [("Purpose", (rating.purpose or "").replace("_", " ").capitalize()),
             ("Your Money or Your Life", f"{(rating.ymyl or '').capitalize()} YMYL"
              + (f" · {rating.ymyl_topic}" if rating.ymyl != "no" and rating.ymyl_topic not in (None, "none") else ""))]
    if rating.needs_met:
        facts.append(("Needs Met for your query", rating.needs_met))
    fact_html = "".join(_figure(label, value, INK, small=True) for label, value in facts)

    bars = ""
    for dim in ("trust", "authoritativeness", "expertise", "experience"):
        v = rating.eeat[dim]
        color = INK if dim == "trust" else (BAD if v < 2.5 else MUTED)
        weight = "font-weight:500" if dim == "trust" else ""
        bars += (f'<div class="sc-bar"><span style="font-size:15px;{weight}">{dim.capitalize()}</span>'
                 f'<div class="sc-track"><div style="width:{v / 4 * 100:.0f}%;height:6px;background:{color}"></div></div>'
                 f'<span style="font-size:14px;color:#3A404A;text-align:right">{_level_name(v)}</span></div>')

    rep = snapshot.reputation
    found = [n for n, v in (("About", rep.about), ("Contact", rep.contact), ("Privacy", rep.privacy),
                            ("Terms", rep.terms)) if v]
    responsible = (f"{', '.join(found)} pages found." if found else "No About or Contact pages linked.")
    responsible += " Editorial policy linked." if rep.editorial_policy else " No editorial policy linked."
    rows = [
        ("Author", (snapshot.author_name or "Not named") + (", with author page" if snapshot.author_url else "")),
        ("Published / updated", f"{(snapshot.published or '–')[:10]} / {(snapshot.modified or '–')[:10]}"),
        ("Schema", " · ".join(snapshot.schema_types) or "None"),
        ("Sources cited", f"{snapshot.external_link_count} outbound links"),
        ("Who's responsible", responsible),
        ("Ads", f"{snapshot.ads.ad_slot_count} ad slots, {snapshot.ads.affiliate_link_count} affiliate links"),
        ("Automatic overrides", "; ".join(rating.gates) or "None triggered"),
    ]
    dl = "".join(f'<dt>{esc(k)}</dt><dd class="{"sc-mono" if k == "Schema" else ""}">{esc(v)}</dd>' for k, v in rows)

    _html(
        f'<div class="sc"><div class="sc-scale" role="img" aria-label="Rated {esc(rating.band)} on a 9-point scale">{cells}</div>'
        f'<div class="sc-scale-l">{labels}</div>'
        f'<div class="sc-figs" style="grid-template-columns:repeat({len(facts)},minmax(0,1fr));border-top-color:{RULE}">{fact_html}</div>'
        f'<section class="sc-sec"><div><h2>E-E-A-T</h2><p class="sc-note">Trust counts most. Experience matters only '
        f'when the topic calls for it.</p></div><div>{bars}</div></section>'
        f'<section class="sc-sec"><h2>What the rater sees</h2><dl class="sc-dl">{dl}</dl></section></div>'
    )

    if s.explanation:
        items = "".join(
            f'<dt>{esc(QUESTIONS_BY_ID[qid].id.replace("_", " ").capitalize())}</dt><dd>{esc(ans.evidence)}</dd>'
            for qid, ans in s.explanation.items() if ans.evidence)
        _html(f'<div class="sc"><section class="sc-sec"><h2>Why, with evidence</h2><dl class="sc-dl">{items}</dl>'
              f'</section></div>')
    backend = "Claude" if any(v == "claude" for v in rating.provenance.values()) else "Jev"
    _html(f'<div class="sc"><p class="sc-note" style="margin-top:40px">Rated by {backend}. '
          f'"Explain this rating" asks Claude for quoted evidence (about $0.01). Rubric {esc(rating.rubric_version)}.</p></div>')


# =============================================================================
# LLM Visibility tab
# =============================================================================

def render_visibility_tab() -> None:
    """Can AI reach it, does AI cite it, probes (on click), crawler table."""
    if _no_analysis():
        return
    a, s = _state().analysis, _state()
    access = a["access"]
    if access is None:
        st.error(a.get("error") or "AI access check did not run.")
        return

    blocked_search = access.search_bots_blocked
    if access.noindex:
        reach, reach_cap = "No", "The page is set to noindex."
    elif blocked_search:
        reach, reach_cap = "Partly", f"robots.txt blocks {', '.join(blocked_search)}."
    elif access.client_rendered_suspect:
        reach, reach_cap = "Barely", "The content appears to be built by JavaScript; most AI crawlers won't see it."
    else:
        training = [b for b in access.blocked_bots if b not in SEARCH_CRITICAL_BOTS]
        reach = "Yes"
        reach_cap = "All AI search crawlers are allowed."
        if training:
            reach_cap += (f" {len(training)} training crawlers are blocked, which limits what future models learn "
                          "but not what they cite today.")

    probes = s.probes
    if probes is not None and probes.completed:
        k, n = probes.cited_count, len(probes.completed)
        sizes = [len(r.cited_urls) for r in probes.completed]
        cite_num = f"{k} of {n} answers"
        cite_cap = (f"{PROBE_ENGINES[DEFAULT_ENGINE]['label']}. Each answer cited {min(sizes)} to {max(sizes)} sources."
                    if sizes else "")
        cite_color = BAD if k == 0 else INK
    else:
        cite_num, cite_cap, cite_color = "Not checked", "Run the probes below to find out.", MUTED

    _html(f'<div class="sc"><div class="sc-figs" style="grid-template-columns:repeat(2,minmax(0,1fr));border-top:0;margin-top:8px">'
          f'{_figure("Can AI reach it?", reach, BAD if reach != "Yes" else INK, caption=reach_cap, small=True)}'
          f'{_figure("Does AI cite it?", cite_num, cite_color, caption=cite_cap, small=True)}</div></div>')

    _html('<div class="sc"><section class="sc-sec" style="margin-top:56px"><h2>Citation probes</h2><div></div></section></div>')
    keys = _keys()
    q1, q2 = st.columns([3, 1], vertical_alignment="bottom")
    with q1:
        st.text_area(f"Questions, one per line (max {MAX_PROBE_QUERIES})", key="probe_queries", height=110)
    queries = [q for q in s.probe_queries.splitlines() if q.strip()]
    with q2:
        st.caption(f"{len(queries)} × {PROBE_ENGINES[DEFAULT_ENGINE]['label']} · about ${estimate_cost(len(queries)):.2f}")
        run = st.button("Run probes", disabled=not (keys and keys.has_openrouter) or not queries,
                        use_container_width=True,
                        help=None if keys and keys.has_openrouter else "Needs OPENROUTER_API_KEY")
    if run and guard("probe"):
        with st.spinner(f"Asking {len(queries)} questions..."):
            try:
                s.probes = run_probes(queries, a["snapshot"].final_url, OpenRouterClient(api_key=keys.openrouter))
                st.rerun()
            except (ProbeError, LLMError) as e:
                st.error(str(e))

    if probes is not None:
        rows = ""
        for r in probes.results:
            if r.error:
                status, firsts = f'<span style="color:{BAD}">Failed</span>', esc(r.error)
            else:
                status = (f'Cited #{r.position}' if r.domain_cited else f'<span style="color:{BAD}">Not cited</span>')
                seen, firsts_list = set(), []
                for h in r.cited_hosts:
                    if h not in seen:
                        seen.add(h)
                        firsts_list.append(h)
                firsts = esc(", ".join(firsts_list[:4]))
            rows += (f'<tr><td>{esc(r.query)}</td><td>{status}</td>'
                     f'<td class="sc-mono" style="color:#3A404A">{firsts}</td></tr>')
        competitors = probes.top_competitors(1)
        note = (f"{esc(competitors[0][0])} appears in {competitors[0][1]} of {len(probes.completed)} answers. "
                if competitors else "")
        cost = f"Cost ${probes.cost:.3f}. " if probes.cost is not None else ""
        _html(f'<div class="sc"><table class="sc-table"><thead><tr><th style="width:36%">Query</th>'
              f'<th style="width:14%">This domain</th><th>Cited first</th></tr></thead><tbody>{rows}</tbody></table>'
              f'<p class="sc-note">{note}{cost}Probes use OpenRouter\'s search, which approximates but does not equal '
              f'the consumer apps.</p></div>')

    bot_rows = "".join(
        f'<tr><td class="sc-mono">{esc(bot)}</td><td style="color:#3A404A">{esc(purpose)}</td>'
        f'<td style="text-align:right;{"color:" + MUTED if access.bot_access[bot] is False else ""}">'
        f'{"Unknown" if access.bot_access[bot] is None else ("Allowed" if access.bot_access[bot] else "Blocked")}</td></tr>'
        for bot, purpose in sorted(AI_BOTS.items(), key=lambda kv: (access.bot_access[kv[0]] is False, kv[0] not in SEARCH_CRITICAL_BOTS)))
    other = [
        ("Indexing", ("Noindex" if access.noindex else "Indexable") + (", snippets blocked" if access.nosnippet else ", snippets allowed")),
        ("Rendering", "Likely JavaScript-built" if access.client_rendered_suspect else "Content is in the server HTML"),
        ("Schema", " · ".join(access.schema_types) or "None"),
        ("llms.txt", "Found" if access.llms_txt_present else "Not found (low impact today)"),
    ]
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in other)
    _html(f'<div class="sc"><section class="sc-sec"><h2>AI crawlers</h2><table class="sc-table"><thead><tr>'
          f'<th>Crawler</th><th>Used for</th><th style="text-align:right">robots.txt</th></tr></thead>'
          f'<tbody>{bot_rows}</tbody></table></section>'
          f'<section class="sc-sec" style="margin-top:48px"><h2>Other signals</h2><dl class="sc-dl">{dl}</dl></section></div>')


# =============================================================================
# Site Audit tab
# =============================================================================

def render_site_audit_tab() -> None:
    """Sitemap sample → rated grid (sortable via st.dataframe) + CSV export."""
    s = _state()
    c1, c4, c2, c3 = st.columns([4.5, 2.2, 2.5, 1.3], vertical_alignment="bottom")
    with c1:
        default_site = s.analysis["snapshot"].host if s.analysis and s.analysis.get("snapshot") else ""
        site = st.text_input("Site", value=default_site, placeholder="example.com", key="site_input")
    with c4:
        scope = st.selectbox("Pages", ["Updated in last 12 months", "Whole site"], key="site_scope",
                             help="Large sites list old archives first. Recent pages reflect the site today.")
    with c2:
        cap = max_site_sample(MAX_SAMPLE)
        n = st.slider("Pages to sample", 5, cap, min(DEFAULT_SAMPLE, cap), step=5)
    with c3:
        go = st.button("Audit site", type="primary", use_container_width=True)

    if go:
        keys = _keys()
        classifier = make_classifier(keys) if keys else None  # Jev by default: one cheap call per page
        if classifier is None:
            st.error("Add TYPESAFE_API_KEY to ~/.config/simcheck/.env to rate pages.")
        elif not site.strip():
            st.warning("Enter a site.")
        elif guard("site_audit"):
            bar = st.progress(0.0, text="Finding pages...")
            try:
                s.site_audit = audit_site(site, classifier, n,
                                          progress=lambda d, t: bar.progress(d / t, text=f"Rated {d} of {t} pages"),
                                          recent_days=None if scope == "Whole site" else 365)
            except SiteAuditError as e:
                st.error(str(e))
            bar.empty()

    audit = s.site_audit
    if audit is None:
        _html('<div class="sc sc-empty">Enter a site and press Audit site. Pages are sampled from its sitemap '
              'and rated with Jev (no Claude cost).</div>')
        return

    summ = audit.summary()
    source = audit.source.split("/")[-1] if audit.source.startswith("http") else audit.source
    _html(f'<div class="sc"><p class="sc-meta">{summ["sampled"]} pages sampled from {esc(audit.pool)} '
          f'({audit.discovered:,} found, starting at {esc(source)}) · rated by Jev · no Claude cost</p>'
          f'<h1 class="sc-h1-sm">{esc(site_headline(summ))}</h1></div>')
    if summ.get("rated"):
        r = summ["rated"]
        figs = [("Rated High or better", f'{summ["high_or_better"]} of {r}'),
                ("Average PQ", f'{summ["avg_pq"]:.0f}'),
                ("Clearly YMYL", f'{summ["clearly_ymyl"]} of {r}'),
                ("Named author", f'{summ["named_author"]} of {r}')]
        _html('<div class="sc"><div class="sc-figs" style="grid-template-columns:repeat(4,minmax(0,1fr))">'
              + "".join(_figure(lbl, val, INK, small=True) for lbl, val in figs) + "</div></div>")

    _html('<div class="sc" style="margin-top:48px"></div>')
    df = pd.DataFrame([{
        "Page": p.path, "Band": p.band, "PQ": None if p.pq_score is None else round(p.pq_score),
        "Trust": None if p.trust is None else round(p.trust, 1), "YMYL": (p.ymyl or "").capitalize(),
        "Author": p.author or "", "Updated": p.updated or "", "Words": p.words,
        "Issue": "; ".join(p.gates) or (p.error or ""),
    } for p in audit.pages]).sort_values("PQ", na_position="first")
    st.dataframe(df, hide_index=True, use_container_width=True, height=min(38 * (len(df) + 1) + 4, 720),
                 column_config={
                     "PQ": st.column_config.NumberColumn(format="%d"),
                     "Trust": st.column_config.NumberColumn(format="%.1f"),
                     "Words": st.column_config.NumberColumn(format="%d"),
                 })
    host = audit.site.split("//")[-1]
    st.download_button("Download CSV", audit_csv(audit), file_name=f"{host}-site-audit.csv", mime="text/csv")
    st.caption("Click a column header to sort.")
