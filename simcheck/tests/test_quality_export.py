"""Tests for exports (simcheck.quality.export). No network."""

import csv
import io
import json
from datetime import date

from simcheck.quality.export import analysis_json, audit_csv, band_color, share_report_html
from simcheck.quality.probe import ProbeReport, evaluate_answer
from simcheck.quality.report import Fix, PageReport
from simcheck.quality.site import SiteAudit, SitePage


URL = "https://example.com/guides/dkim"


def _report(headline="Google would rate this page High+. AI search engines aren't citing it."):
    return PageReport(
        headline=headline,
        summary=("The page clears Google's bar.", "Every AI search crawler can reach it."),
        fixes=(Fix("Answer first", "Define it up top.", 15, "content"), Fix("Add FAQ", "Use lost queries.", 25, "content")),
        quality_line="", visibility_line="", simscore_line="",
    )


def _probes():
    results = tuple(evaluate_answer(f"q{i}", "perplexity", "", ["https://mayoclinic.org/a"], URL, "example")
                    for i in range(3))
    return ProbeReport(URL, "example.com", "example", results, 0.02)


class _Rating:
    rated = True
    band = "High+"


def test_share_report_contents():
    html = share_report_html(URL, _report(), _Rating(), _probes(), report_date=date(2026, 10, 1))
    assert html.startswith("<!doctype html>")
    assert "1 October 2026" in html and "example.com" in html
    assert "High+" in html and "Second-highest on Google" in html
    assert "0 of 3" in html and "#B4541A" in html
    assert "<li" in html and "about 40 minutes" in html
    assert "Perplexity Sonar" in html
    assert "<script" not in html.lower()
    for leaked in ("confidence", "probabilit", "Jev", "Claude"):
        assert leaked not in html


def test_share_report_escapes_untrusted_text():
    evil = '<img src=x onerror="alert(1)">'
    html = share_report_html(URL + "?" + evil, _report(headline=evil), None, None, prepared_for=evil)
    assert "<img" not in html and "&lt;img" in html


def test_share_report_without_rating_or_probes():
    html = share_report_html(URL, _report(), None, None)
    assert "Google page quality" not in html and "Cited in AI answers" not in html
    assert "Perplexity" not in html


def test_analysis_json_round_trips():
    data = json.loads(analysis_json(URL, "dkim", report=_report(), probes=_probes(), rating=None))
    assert data["url"] == URL and data["query"] == "dkim"
    assert "rating" not in data
    assert data["report"]["fixes"][0]["title"] == "Answer first"
    assert data["probes"]["results"][0]["cited_hosts"] == ["mayoclinic.org"]


def test_audit_csv():
    page = SitePage(URL, "/guides/dkim", "DKIM", True, "High", 3.0, 74.6, "no", 3.04, "Jane", "2026-08-01", 512, ())
    text = audit_csv(SiteAudit("https://example.com", "sitemap", 1, (page,)))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["band"] == "High" and rows[0]["pq_score"] == "75" and rows[0]["trust"] == "3.0"
    assert audit_csv(SiteAudit("https://example.com", "", 0, ())) == ""


def test_band_color():
    assert band_color("High+") == "#1F5BBF"
    assert band_color("Nonsense") == "#15181D"
