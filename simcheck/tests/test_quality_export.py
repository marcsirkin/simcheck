"""Tests for exports (simcheck.quality.export). No network."""

import csv
import io
import json
from datetime import date
from types import SimpleNamespace

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
    pq_score_rounded = 88
    purpose = "informational"
    ymyl = "no"
    ymyl_topic = "none"
    needs_met = "Highly Meets"
    eeat = {"trust": 3.5, "authoritativeness": 3.0, "expertise": 3.5, "experience": 2.5}
    reasons = ("Trust rated High.",)
    gates = ()


def test_share_report_contents():
    html = share_report_html(URL, _report(), _Rating(), _probes(), report_date=date(2026, 10, 1))
    assert html.startswith("<!doctype html>")
    assert "1 October 2026" in html and "example.com" in html
    assert "High+" in html and "Second-highest on Google" in html
    assert "0 of 3" in html and "#B4541A" in html
    assert "<li" in html and "about 40 minutes" in html
    assert "Perplexity Sonar" in html
    assert "<script" not in html.lower()
    # Claude crawler names are client-facing access facts; model provenance is not.
    for leaked in ("confidence", "probabilit", "Jev", "Rated by Claude"):
        assert leaked not in html


def test_share_report_escapes_untrusted_text():
    evil = '<img src=x onerror="alert(1)">'
    html = share_report_html(URL + "?" + evil, _report(headline=evil), None, None, prepared_for=evil)
    assert "<img" not in html and "&lt;img" in html


def test_share_report_without_rating_or_probes():
    html = share_report_html(URL, _report(), None, None)
    assert "Google page quality" not in html and "Cited in AI answers" not in html
    assert "Perplexity" not in html


def test_share_report_includes_all_available_sections():
    snapshot = SimpleNamespace(
        title="DKIM guide", author_name="Jane", published="2026-01-01", modified="2026-09-01",
        lang="en", word_count=1200, headings=(("h1", "DKIM"), ("h2", "Setup")), h1_count=1,
        schema_types=("Article",), external_link_count=4,
        reputation=SimpleNamespace(about="/about", contact="/contact", privacy=None, terms=None,
                                   editorial_policy="/standards"),
        ads=SimpleNamespace(ad_slot_count=0, affiliate_link_count=0, sponsored_link_count=0),
    )
    access = SimpleNamespace(
        bot_access={name: (name != "GPTBot") for name in (
            "GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "Claude-SearchBot",
            "Claude-User", "PerplexityBot", "Google-Extended", "CCBot",
        )},
        noindex=False, nosnippet=False, client_rendered_suspect=False,
        schema_types=("Article",), llms_txt_present=True,
        issues=(SimpleNamespace(severity="low", message="Training crawler blocked."),),
    )
    readiness = SimpleNamespace(
        score_rounded=72, interpretation="Solid signal coverage",
        components={"coverage": 66, "structure": 80, "evidence": 70, "answerability": 75},
    )
    geo = SimpleNamespace(
        intent=SimpleNamespace(value="informational"), page_type=SimpleNamespace(value="article"),
        steps=(SimpleNamespace(title="Clarify the opening", priority=SimpleNamespace(value="medium"),
                               minutes=15, why="The answer appears late.",
                               how="Move the clearest answer earlier if it reads naturally."),),
    )
    chunks = [SimpleNamespace(chunk_index=0, similarity=0.54, normalized_score=1.0,
                              interpretation="Weak", text="DKIM authenticates outbound email.")]
    diagnostic = SimpleNamespace(
        query="how to configure DKIM",
        coverage=SimpleNamespace(score_rounded=58, interpretation="Weak"),
        summary=SimpleNamespace(total_chunks=1, min_similarity=0.54, max_similarity=0.54,
                                avg_similarity=0.54, chunks_strong=0, chunks_moderate=0,
                                chunks_weak=1, chunks_off_topic=0),
        by_document_order=lambda: chunks,
    )
    explanation = {"trust": SimpleNamespace(evidence="The named author links sources.")}

    html = share_report_html(
        URL, _report(), _Rating(), _probes(), snapshot=snapshot, access=access,
        readiness=readiness, geo=geo, diagnostic=diagnostic, explanation=explanation,
        report_date=date(2026, 10, 1),
    )

    for expected in (
        "Page Quality", "E-E-A-T", "What the rater saw", "Evidence explanation",
        "LLM Visibility", "Crawler access", "Citation probes", "Content Match",
        "Experimental content patterns", "All editorial options", "Chunk diagnostics",
        "DKIM authenticates outbound email.",
    ):
        assert expected in html
    for leaked in ("confidence", "probabilit", "Jev", "Rated by Claude"):
        assert leaked not in html


def test_analysis_json_round_trips():
    data = json.loads(analysis_json(
        URL, "dkim", report=_report(), probes=_probes(), rating=None,
        snapshot={"title": "DKIM guide"}, geo={"page_type": "article"},
        diagnostic={"chunks": [{"similarity": 0.54}]}, explanation={"trust": "evidence"},
    ))
    assert data["url"] == URL and data["query"] == "dkim"
    assert "rating" not in data
    assert data["report"]["fixes"][0]["title"] == "Answer first"
    assert data["probes"]["results"][0]["cited_hosts"] == ["mayoclinic.org"]
    assert data["snapshot"]["title"] == "DKIM guide"
    assert data["geo"]["page_type"] == "article"
    assert data["diagnostic"]["chunks"][0]["similarity"] == 0.54
    assert data["explanation"]["trust"] == "evidence"


def test_audit_csv():
    page = SitePage(URL, "/guides/dkim", "DKIM", True, "High", 3.0, 74.6, "no", 3.04, "Jane", "2026-08-01", 512, ())
    text = audit_csv(SiteAudit("https://example.com", "sitemap", 1, (page,)))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["band"] == "High" and rows[0]["pq_score"] == "75" and rows[0]["trust"] == "3.0"
    assert audit_csv(SiteAudit("https://example.com", "", 0, ())) == ""


def test_band_color():
    assert band_color("High+") == "#1F5BBF"
    assert band_color("Nonsense") == "#15181D"
