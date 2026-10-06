"""
Exports: shareable client report (HTML), full analysis (JSON), site grid (CSV).

The shareable report is a single self-contained HTML file: no scripts, no
probabilities or model names, plain-language findings only. Streamlit
can't host a public page, so the report is downloaded and shared as a
file (or opened and printed to PDF).

All page-derived text is HTML-escaped: titles and URLs come from
third-party sites and must never be able to inject markup.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, is_dataclass
from datetime import date
from enum import Enum
from html import escape
from typing import Optional

from simcheck.core.diagnostics import DiagnosticReport
from simcheck.core.geo import GeoNextStepsReport
from simcheck.core.readiness import ReadinessScore
from simcheck.quality.ai_access import AIAccessReport, AI_BOTS, SEARCH_CRITICAL_BOTS
from simcheck.quality.page_quality import PageQualityRating
from simcheck.quality.probe import ProbeReport
from simcheck.quality.report import PageReport
from simcheck.quality.site import SiteAudit, audit_to_rows
from simcheck.quality.snapshot import PageSnapshot


BAND_COLORS = {
    "Lowest": "#9A3412", "Lowest+": "#9A3412", "Low": "#B4541A", "Low+": "#B4541A",
    "Medium": "#4B5563", "Medium+": "#4B5563", "High": "#1F5BBF", "High+": "#1F5BBF",
    "Highest": "#173F8A", "Unrated": "#4B5563",
}
CITE_BAD = "#B4541A"
INK = "#15181D"

_SLIDER_POSITION_TEXT = {
    "Lowest": "the lowest rating", "Lowest+": "second-lowest", "Low": "third from the bottom",
    "Low+": "below the midpoint", "Medium": "the midpoint", "Medium+": "above the midpoint",
    "High": "third from the top", "High+": "second-highest", "Highest": "the highest rating",
}


def band_color(band: str) -> str:
    """Display color for a QRG band."""
    return BAND_COLORS.get(band, INK)


def _p(text: str) -> str:
    return f'<p style="margin:14px 0 0;font-size:16px;line-height:1.65;color:#2A2F37">{escape(text)}</p>'


def _section(title: str, body: str) -> str:
    """One complete-report section, omitted when it has no content."""
    if not body:
        return ""
    return (
        f'<section class="report-section"><h2>{escape(title)}</h2>'
        f'<div class="section-body">{body}</div></section>'
    )


def _definition_list(rows: list[tuple[str, object]]) -> str:
    if not rows:
        return ""
    items = "".join(
        f'<dt>{escape(str(label))}</dt><dd>{escape("" if value is None else str(value))}</dd>'
        for label, value in rows
    )
    return f'<dl class="details">{items}</dl>'


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape('' if value is None else str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _level_name(value: object) -> str:
    try:
        level = int(min(max(float(value), 0), 4) + 0.5)
    except (TypeError, ValueError):
        return "Not available"
    return ("Lowest", "Low", "Medium", "High", "Highest")[level]


def _page_quality_section(
    rating: Optional[PageQualityRating],
    snapshot: Optional[PageSnapshot],
    explanation: Optional[dict],
) -> str:
    if rating is None:
        return ""
    if not rating.rated:
        reasons = "".join(f"<li>{escape(str(reason))}</li>" for reason in rating.reasons)
        return _section("Page Quality", f"<p>The page was not rated.</p><ul>{reasons}</ul>")

    rows = [
        ("Rating", f"{rating.band} ({rating.pq_score_rounded}/100)"),
        ("Page purpose", (rating.purpose or "Not available").replace("_", " ").capitalize()),
        ("YMYL", (rating.ymyl or "Not available").capitalize()),
        ("YMYL topic", rating.ymyl_topic or "Not applicable"),
        ("Needs Met", rating.needs_met or "Not evaluated"),
    ]
    body = _definition_list(rows)

    if rating.eeat:
        eeat_rows = [
            (dimension.capitalize(), _level_name(rating.eeat.get(dimension)), f"{rating.eeat.get(dimension, 0):.2f} / 4")
            for dimension in ("trust", "authoritativeness", "expertise", "experience")
        ]
        body += '<h3>E-E-A-T</h3>' + _table(("Dimension", "Level", "Score"), eeat_rows)

    if rating.reasons:
        body += '<h3>Rating drivers</h3><ul>' + "".join(
            f"<li>{escape(str(reason))}</li>" for reason in rating.reasons
        ) + "</ul>"
    if rating.gates:
        body += '<h3>Automatic overrides</h3><ul>' + "".join(
            f"<li>{escape(str(gate))}</li>" for gate in rating.gates
        ) + "</ul>"

    if snapshot is not None:
        rep = snapshot.reputation
        responsible = [name for name, value in (
            ("About", rep.about), ("Contact", rep.contact), ("Privacy", rep.privacy),
            ("Terms", rep.terms), ("Editorial policy", rep.editorial_policy),
        ) if value]
        snapshot_rows = [
            ("Title", snapshot.title or "Not found"),
            ("Author", snapshot.author_name or "Not named"),
            ("Published", (snapshot.published or "Not found")[:10]),
            ("Updated", (snapshot.modified or "Not found")[:10]),
            ("Language", snapshot.lang or "Not declared"),
            ("Words", f"{snapshot.word_count:,}"),
            ("Headings", f"{len(snapshot.headings)} ({snapshot.h1_count} H1)"),
            ("Schema", ", ".join(snapshot.schema_types) or "None"),
            ("External sources", f"{snapshot.external_link_count} links"),
            ("Responsibility pages", ", ".join(responsible) or "None linked"),
            ("Monetization", (
                f"{snapshot.ads.ad_slot_count} ad slots; {snapshot.ads.affiliate_link_count} affiliate links; "
                f"{snapshot.ads.sponsored_link_count} sponsored links"
            )),
        ]
        body += '<h3>What the rater saw</h3>' + _definition_list(snapshot_rows)

    evidence = []
    for question_id, answer in (explanation or {}).items():
        text = getattr(answer, "evidence", None)
        if text:
            evidence.append((question_id.replace("_", " ").capitalize(), text))
    if evidence:
        body += '<h3>Evidence explanation</h3>' + _definition_list(evidence)
    return _section("Page Quality", body)


def _visibility_section(access: Optional[AIAccessReport], probes: Optional[ProbeReport]) -> str:
    if access is None and probes is None:
        return ""
    body = ""
    if access is not None:
        bot_rows = []
        for bot, purpose in sorted(AI_BOTS.items()):
            value = access.bot_access.get(bot)
            status = "Unknown" if value is None else ("Allowed" if value else "Blocked")
            surface = "AI search or live retrieval" if bot in SEARCH_CRITICAL_BOTS else "Training or grounding"
            bot_rows.append((bot, purpose, surface, status))
        body += '<h3>Crawler access</h3>' + _table(
            ("Crawler", "Used for", "Impact", "robots.txt"), bot_rows
        )
        body += '<h3>Other access signals</h3>' + _definition_list([
            ("Indexing", "Noindex" if access.noindex else "Indexable"),
            ("Snippets", "Blocked" if access.nosnippet else "Allowed"),
            ("Rendering", "Likely JavaScript-built" if access.client_rendered_suspect else "Content in server HTML"),
            ("Schema", ", ".join(access.schema_types) or "None"),
            ("llms.txt", "Found" if access.llms_txt_present else "Not found"),
        ])
        if access.issues:
            body += '<h3>Access findings</h3><ul>' + "".join(
                f"<li><strong>{escape(issue.severity.capitalize())}:</strong> {escape(issue.message)}</li>"
                for issue in access.issues
            ) + "</ul>"

    if probes is not None:
        probe_rows = []
        for result in probes.results:
            if result.error:
                status = f"Failed: {result.error}"
            elif result.domain_cited:
                status = f"Cited at position {result.position}" if result.position else "Domain cited"
            else:
                status = "Not cited"
            probe_rows.append((result.query, status, ", ".join(dict.fromkeys(result.cited_hosts)) or "None"))
        body += '<h3>Citation probes</h3>' + _table(("Query", "This domain", "Cited domains"), probe_rows)
        competitors = probes.top_competitors(10)
        if competitors:
            body += '<h3>Frequently cited alternatives</h3>' + _table(
                ("Domain", "Answers citing it"), [(host, count) for host, count in competitors]
            )
        if probes.cost is not None:
            body += f'<p class="note">Probe cost: ${probes.cost:.3f}. Results can vary between runs.</p>'
    return _section("LLM Visibility", body)


def _content_match_section(
    readiness: Optional[ReadinessScore],
    geo: Optional[GeoNextStepsReport],
    diagnostic: Optional[DiagnosticReport],
) -> str:
    if readiness is None and geo is None and diagnostic is None:
        return ""
    body = ""
    if diagnostic is not None:
        coverage = diagnostic.coverage
        summary = diagnostic.summary
        body += _definition_list([
            ("Target query", diagnostic.query),
            ("Concept Coverage Score", f"{coverage.score_rounded}/100 ({coverage.interpretation})"),
            ("Chunks", summary.total_chunks),
            ("Raw cosine range", f"{summary.min_similarity:.3f} to {summary.max_similarity:.3f}"),
            ("Mean raw cosine", f"{summary.avg_similarity:.3f}"),
            ("Provisional bands", (
                f"{summary.chunks_strong} strong; {summary.chunks_moderate} moderate; "
                f"{summary.chunks_weak} weak; {summary.chunks_off_topic} off-topic"
            )),
        ])
    if readiness is not None:
        body += '<h3>Experimental content patterns</h3>' + _definition_list([
            ("Score", f"{readiness.score_rounded}/100 ({readiness.interpretation})"),
            ("Coverage", f"{readiness.components['coverage']:.0f}/100"),
            ("Structure", f"{readiness.components['structure']:.0f}/100"),
            ("Evidence", f"{readiness.components['evidence']:.0f}/100"),
            ("Opening clarity", f"{readiness.components['answerability']:.0f}/100"),
        ])
        body += '<p class="note">This experimental heuristic does not predict whether an AI answer engine will cite the page.</p>'
    if geo is not None:
        body += '<h3>Page context</h3>' + _definition_list([
            ("Query intent", geo.intent.value.replace("_", " ").capitalize()),
            ("Page purpose", geo.page_type.value.replace("_", " ").capitalize()),
        ])
        if geo.steps:
            body += '<h3>All editorial options</h3><ol class="options">' + "".join(
                f'<li><strong>{escape(step.title)}</strong>'
                f'<span class="meta">{escape(step.priority.value.capitalize())} priority · about {step.minutes} minutes</span>'
                f'<p>{escape(step.why)}</p><p>{escape(step.how)}</p></li>'
                for step in geo.steps
            ) + "</ol>"
    if diagnostic is not None:
        chunk_rows = [
            (
                chunk.chunk_index + 1,
                f"{chunk.similarity:.3f}",
                f"{chunk.normalized_score:.2f}",
                f"{chunk.interpretation} (provisional)",
                chunk.text,
            )
            for chunk in diagnostic.by_document_order()
        ]
        body += '<h3>Chunk diagnostics</h3><p class="note">Raw cosine drives the provisional band. Relative position is min-max placement within this page only.</p>'
        body += _table(("#", "Raw cosine", "Relative", "Band", "Analyzed text"), chunk_rows)
    return _section("Content Match", body)


def share_report_html(
    url: str,
    report: PageReport,
    rating: Optional[PageQualityRating],
    probes: Optional[ProbeReport],
    prepared_for: str = "",
    report_date: Optional[date] = None,
    *,
    snapshot: Optional[PageSnapshot] = None,
    access: Optional[AIAccessReport] = None,
    readiness: Optional[ReadinessScore] = None,
    geo: Optional[GeoNextStepsReport] = None,
    diagnostic: Optional[DiagnosticReport] = None,
    explanation: Optional[dict] = None,
) -> str:
    """
    Render the shareable client report.

    Args:
        url: Analyzed page URL
        report: PageReport from build_report()
        rating: Page Quality rating (None if not rated)
        probes: Probe results (None if not run)
        prepared_for: Optional client name for the header
        report_date: Date shown (defaults to today)
        snapshot: Deterministic page metadata and content signals
        access: AI crawler, indexing, rendering, schema, and llms.txt results
        readiness: Experimental content-pattern score and components
        geo: Page context and complete editorial options
        diagnostic: Chunk-level Content Match results
        explanation: Optional evidence generated for the Page Quality rating

    Returns:
        Complete HTML document as a string
    """
    d = (report_date or date.today()).strftime("%-d %B %Y")
    host = url.split("//")[-1].split("/")[0].removeprefix("www.")

    figures = []
    if rating is not None and rating.rated:
        figures.append(
            f'<div style="padding:20px 24px 24px 0"><div style="font-size:13px;color:#5B6270">Google page quality</div>'
            f'<div style="margin-top:6px;font-size:40px;font-weight:600;letter-spacing:-0.02em;color:{band_color(rating.band)}">'
            f'{escape(rating.band)}</div><div style="margin-top:4px;font-size:14px;color:#3A404A">'
            f'{escape(_SLIDER_POSITION_TEXT.get(rating.band, "").capitalize())} on Google\'s 9-point scale</div></div>')
    if probes is not None and probes.completed:
        n, k = len(probes.completed), probes.cited_count
        color = CITE_BAD if k == 0 else INK
        figures.append(
            f'<div style="padding:20px 0 24px 24px;border-left:1px solid #E3E6EA"><div style="font-size:13px;color:#5B6270">'
            f'Cited in AI answers</div><div style="margin-top:6px;font-size:40px;font-weight:600;letter-spacing:-0.02em;'
            f'color:{color}">{k} of {n}</div><div style="margin-top:4px;font-size:14px;color:#3A404A">'
            f'Perplexity, {n} common questions</div></div>')
    figures_html = ""
    if figures:
        cols = len(figures)
        figures_html = (f'<div style="margin-top:40px;display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));'
                        f'border-top:1px solid {INK};border-bottom:1px solid #E3E6EA">{"".join(figures)}</div>')

    found = "".join(_p(par) for par in report.summary)
    fixes = "".join(
        f'<li style="margin-top:10px;padding-left:6px">{escape(f.title)}. <span style="color:#5B6270">'
        f'{escape(f.detail)}</span></li>' for f in report.fixes)
    total_minutes = sum(f.minutes for f in report.fixes if f.minutes)
    effort = _p(f"Estimated review and editing time: about {total_minutes} minutes.") if total_minutes else ""

    method = "Method: page quality follows Google's Search Quality Rater Guidelines (September 2025 edition)."
    if probes is not None and probes.completed:
        method += (f" AI citation checks asked {len(probes.completed)} questions through Perplexity Sonar on {d};"
                   " results vary from run to run.")
    method += " Prepared with SimCheck."

    detail_sections = (
        _page_quality_section(rating, snapshot, explanation)
        + _visibility_section(access, probes)
        + _content_match_section(readiness, geo, diagnostic)
    )

    header_left = escape(prepared_for) if prepared_for else escape(host)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{escape(host)} page report</title>'
        '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&amp;family=Geist+Mono&amp;display=swap" rel="stylesheet">'
        '<style>'
        'body{margin:0;background:#fff}.report-section{display:grid;grid-template-columns:180px minmax(0,1fr);gap:28px;'
        'margin-top:56px;padding-top:24px;border-top:1px solid #E3E6EA}.report-section>h2{margin:0;font-size:17px}'
        '.section-body h3{margin:30px 0 10px;font-size:15px}.section-body h3:first-child{margin-top:0}'
        '.section-body p,.section-body li{font-size:15px;line-height:1.55;color:#2A2F37}.section-body ul{margin:10px 0;padding-left:20px}'
        '.details{display:grid;grid-template-columns:170px minmax(0,1fr);margin:0;border-top:1px solid #E3E6EA;font-size:14px}'
        '.details dt,.details dd{padding:10px 0;border-bottom:1px solid #E3E6EA}.details dt{color:#5B6270}.details dd{margin:0}'
        '.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:#5B6270;'
        'font-weight:500;border-bottom:1px solid #15181D;padding:9px 12px 9px 0}td{vertical-align:top;border-bottom:1px solid #E3E6EA;'
        'padding:10px 12px 10px 0;line-height:1.45}.options{padding-left:20px}.options li{margin:0 0 20px}.options p{margin:5px 0}'
        '.meta{display:block;margin-top:3px;font-size:12px;color:#5B6270}.note{font-size:12.5px!important;color:#5B6270!important}'
        '@media(max-width:720px){.report-section{grid-template-columns:1fr}.details{grid-template-columns:130px minmax(0,1fr)}}'
        '@media print{body{-webkit-print-color-adjust:exact}.report-section{break-inside:auto}tr{break-inside:avoid}}'
        '</style></head>'
        f'<body><article style="font-family:Geist,system-ui,sans-serif;color:{INK};max-width:860px;margin:0 auto;padding:72px 24px 96px">'
        f'<div style="display:flex;justify-content:space-between;gap:16px;font-size:13px;color:#5B6270">'
        f'<span>{header_left}</span><span>{d}</span></div>'
        f'<h1 style="margin:48px 0 0;font-size:34px;line-height:1.18;font-weight:600;letter-spacing:-0.02em">{escape(report.headline)}</h1>'
        f'<p style="margin:14px 0 0;font-size:14px;color:#5B6270;font-family:\'Geist Mono\',monospace;word-break:break-all">{escape(url)}</p>'
        f'{figures_html}'
        f'<h2 style="margin:48px 0 0;font-size:17px;font-weight:600">What we found</h2>{found}'
        + (f'<h2 style="margin:44px 0 0;font-size:17px;font-weight:600">Options to consider</h2>'
           f'<ol style="margin:14px 0 0;padding-left:22px;font-size:16px;line-height:1.65;color:#2A2F37">{fixes}</ol>{effort}'
           if report.fixes else "")
        + detail_sections
        + f'<footer style="margin-top:64px;padding-top:20px;border-top:1px solid #E3E6EA;font-size:12.5px;line-height:1.6;'
          f'color:#5B6270">{escape(method)}</footer></article></body></html>'
    )


def _jsonable(obj):
    """dataclasses / enums / tuples -> JSON-safe structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def analysis_json(url: str, query: Optional[str], **parts) -> str:
    """
    Full structured analysis for archiving and run-to-run comparison.

    Args:
        url: Analyzed URL
        query: Target query
        **parts: Any available analysis objects, including the snapshot,
            report, rating, access, readiness, GEO context, diagnostics,
            probes, and explanation evidence (None values are skipped)

    Returns:
        Pretty-printed JSON string
    """
    data = {"url": url, "query": query, "generated": date.today().isoformat()}
    for name, value in parts.items():
        if value is not None:
            data[name] = _jsonable(value)
    if "rating" in data:
        # Raw answers include long evidence strings; keep them, but drop the
        # duplicated per-answer dicts' empty fields for readability.
        data["rating"]["answers"] = {k: {kk: vv for kk, vv in v.items() if vv not in (None, {}, "")}
                                     for k, v in data["rating"].get("answers", {}).items()}
    return json.dumps(data, indent=2, default=str)


def audit_csv(audit: SiteAudit) -> str:
    """Site audit grid as CSV text."""
    rows = audit_to_rows(audit)
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue()
