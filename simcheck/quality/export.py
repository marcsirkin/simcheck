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

from simcheck.quality.page_quality import PageQualityRating
from simcheck.quality.probe import ProbeReport
from simcheck.quality.report import PageReport
from simcheck.quality.site import SiteAudit, audit_to_rows


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


def share_report_html(
    url: str,
    report: PageReport,
    rating: Optional[PageQualityRating],
    probes: Optional[ProbeReport],
    prepared_for: str = "",
    report_date: Optional[date] = None,
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
    effort = _p(f"These changes take about {total_minutes} minutes of editing.") if total_minutes else ""

    method = "Method: page quality follows Google's Search Quality Rater Guidelines (September 2025 edition)."
    if probes is not None and probes.completed:
        method += (f" AI citation checks asked {len(probes.completed)} questions through Perplexity Sonar on {d};"
                   " results vary from run to run.")
    method += " Prepared with SimCheck."

    header_left = escape(prepared_for) if prepared_for else escape(host)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{escape(host)} page report</title>'
        '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&amp;family=Geist+Mono&amp;display=swap" rel="stylesheet">'
        '<style>body{margin:0;background:#fff}@media print{body{-webkit-print-color-adjust:exact}}</style></head>'
        f'<body><article style="font-family:Geist,system-ui,sans-serif;color:{INK};max-width:680px;margin:0 auto;padding:72px 24px 96px">'
        f'<div style="display:flex;justify-content:space-between;gap:16px;font-size:13px;color:#5B6270">'
        f'<span>{header_left}</span><span>{d}</span></div>'
        f'<h1 style="margin:48px 0 0;font-size:34px;line-height:1.18;font-weight:600;letter-spacing:-0.02em">{escape(report.headline)}</h1>'
        f'<p style="margin:14px 0 0;font-size:14px;color:#5B6270;font-family:\'Geist Mono\',monospace;word-break:break-all">{escape(url)}</p>'
        f'{figures_html}'
        f'<h2 style="margin:48px 0 0;font-size:17px;font-weight:600">What we found</h2>{found}'
        + (f'<h2 style="margin:44px 0 0;font-size:17px;font-weight:600">What to change</h2>'
           f'<ol style="margin:14px 0 0;padding-left:22px;font-size:16px;line-height:1.65;color:#2A2F37">{fixes}</ol>{effort}'
           if report.fixes else "")
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
        **parts: Any of report, rating, access, readiness, probes (None skipped)

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
