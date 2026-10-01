"""Tests for probes and the executive report. No network."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simcheck.core.geo import ContentSignals, GeoIntent, GeoNextStep, GeoNextStepsReport, GeoPriority
from simcheck.core.readiness import ReadinessScore
from simcheck.quality.ai_access import SiteFiles, build_ai_access_report
from simcheck.quality.classifier import FakeClassifier
from simcheck.quality.llm_client import LLMError, OpenRouterClient
from simcheck.quality.page_quality import rate_page
from simcheck.quality.probe import (
    MAX_PROBE_QUERIES,
    ProbeError,
    ProbeReport,
    brand_from_domain,
    default_queries,
    estimate_cost,
    evaluate_answer,
    normalize_host,
    run_probes,
)
from simcheck.quality.report import build_headline, build_report, rank_fixes, Fix
from simcheck.quality.snapshot import parse_snapshot


FIXTURES = Path(__file__).parent / "fixtures"
TARGET = "https://www.healthline.com/health/high-blood-pressure-hypertension"


# =============================================================================
# Probes
# =============================================================================

class TestProbeHelpers:
    @pytest.mark.parametrize("raw,host", [("https://www.Mayo.org/x", "mayo.org"), ("www.cdc.gov", "cdc.gov"),
                                          ("https://my.clevelandclinic.org:443/a", "my.clevelandclinic.org")])
    def test_normalize_host(self, raw, host):
        assert normalize_host(raw) == host

    @pytest.mark.parametrize("domain,brand", [("healthline.com", "healthline"), ("www.bbc.co.uk", "bbc"),
                                              ("blog.semrush.com", "semrush"), ("localhost", "localhost")])
    def test_brand_from_domain(self, domain, brand):
        assert brand_from_domain(domain) == brand

    def test_default_queries(self):
        assert default_queries("high blood pressure") == [
            "high blood pressure", "What is high blood pressure?", "What should I know about high blood pressure?"]
        assert default_queries("how do I set up DKIM?") == ["how do I set up DKIM?"]
        assert default_queries(None, "DKIM Explained | Example Mail") == [
            "DKIM Explained", "What is DKIM Explained?", "What should I know about DKIM Explained?"]
        assert default_queries("", "") == []

    def test_estimate_cost(self):
        assert estimate_cost(3) == pytest.approx(0.018)


class TestEvaluateAnswer:
    def test_domain_cited_with_position(self):
        r = evaluate_answer("q", "perplexity", "Per Healthline, ...",
                            ["https://mayoclinic.org/a", "https://healthline.com/health/other"], TARGET, "healthline")
        assert r.domain_cited and r.position == 2 and not r.url_cited and r.brand_mentioned

    def test_exact_url_match_ignores_www_and_trailing_slash(self):
        r = evaluate_answer("q", "perplexity", "", ["https://healthline.com/health/high-blood-pressure-hypertension/"],
                            TARGET, "healthline")
        assert r.url_cited and r.position == 1

    def test_subdomain_counts_as_domain(self):
        r = evaluate_answer("q", "perplexity", "", ["https://news.healthline.com/x"], TARGET, "healthline")
        assert r.domain_cited

    def test_lookalike_domain_does_not_count(self):
        r = evaluate_answer("q", "perplexity", "", ["https://nothealthline.com/x"], TARGET, "healthline")
        assert not r.domain_cited

    def test_not_cited(self):
        r = evaluate_answer("q", "perplexity", "Mayo says...", ["https://www.mayoclinic.org/a"], TARGET, "healthline")
        assert not r.domain_cited and r.position is None and not r.brand_mentioned
        assert r.cited_hosts == ("mayoclinic.org",)


class _FakeProbeLLM:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def chat_with_citations(self, model, prompt, max_tokens=600):
        self.asked.append(prompt)
        a = self.answers[len(self.asked) - 1]
        if isinstance(a, Exception):
            raise a
        return a


class TestRunProbes:
    def test_report_counts_and_competitors(self):
        llm = _FakeProbeLLM([
            ("text", ["https://mayoclinic.org/a", "https://who.int/b"], 0.005),
            ("text", ["https://mayoclinic.org/c", "https://healthline.com/x"], 0.006),
        ])
        rep = run_probes(["q1", " ", "q2"], TARGET, llm)
        assert llm.asked == ["q1", "q2"]
        assert rep.cited_count == 1 and rep.brand == "healthline"
        assert rep.top_competitors() == [("mayoclinic.org", 2), ("who.int", 1)]
        assert rep.cost == pytest.approx(0.011)

    def test_partial_failure_recorded(self):
        llm = _FakeProbeLLM([LLMError("rate limited"), ("t", ["https://cdc.gov/x"], None)])
        rep = run_probes(["q1", "q2"], TARGET, llm)
        assert len(rep.results) == 2 and len(rep.completed) == 1
        assert rep.results[0].error == "rate limited"
        assert rep.cost is None

    def test_all_fail_raises(self):
        with pytest.raises(ProbeError, match="All probes failed"):
            run_probes(["q1"], TARGET, _FakeProbeLLM([LLMError("down")]))

    @pytest.mark.parametrize("queries,match", [([], "at least one"), (["  "], "at least one"),
                                               (["q"] * (MAX_PROBE_QUERIES + 1), "At most")])
    def test_input_validation(self, queries, match):
        with pytest.raises(ProbeError, match=match):
            run_probes(queries, TARGET, _FakeProbeLLM([]))

    def test_unknown_engine(self):
        with pytest.raises(ProbeError, match="engine"):
            run_probes(["q"], TARGET, _FakeProbeLLM([]), engine="bing")


class TestChatWithCitations:
    def test_collects_url_citations_deduped(self):
        message = SimpleNamespace(content="answer", annotations=[
            {"type": "url_citation", "url_citation": {"url": "https://a.com/1"}},
            {"type": "url_citation", "url_citation": {"url": "https://a.com/1"}},
            {"type": "file", "file": {}},
            {"type": "url_citation", "url_citation": {"url": "https://b.com/2"}},
        ])
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(cost=0.005))
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
        text, urls, cost = OpenRouterClient(client=fake).chat_with_citations("perplexity/sonar", "q")
        assert text == "answer" and urls == ["https://a.com/1", "https://b.com/2"] and cost == 0.005


# =============================================================================
# Report
# =============================================================================

def _snapshot():
    html = (FIXTURES / "good_article.html").read_text()
    filler = "<p>" + "DKIM signatures protect message integrity in transit. " * 10 + "</p>"
    return parse_snapshot("https://example.com/guides/dkim", html.replace("</article>", filler + "</article>"))


def _access(snapshot, robots="User-agent: *\nAllow: /\n"):
    return build_ai_access_report(snapshot, SiteFiles(robots, 200, False, 404))


def _readiness(score, **components):
    comps = {"coverage": 52.0, "structure": 65.0, "evidence": 60.0, "answerability": 72.0, **components}
    return ReadinessScore(score=score, components=comps, weights={}, interpretation="")


def _probes(cited_flags):
    results = tuple(evaluate_answer(f"q{i}", "perplexity", "",
                                    ["https://example.com/x"] if c else ["https://mayoclinic.org/a", "https://cdc.gov/b"],
                                    "https://example.com/guides/dkim", "example")
                    for i, c in enumerate(cited_flags))
    return ProbeReport("https://example.com/guides/dkim", "example.com", "example", results, 0.01)


def _geo():
    steps = [
        GeoNextStep("Front-load a direct answer + definition (first ~150 words)", GeoPriority.HIGH, "why",
                    "Open with a 2-sentence definition.\nThen list symptoms.", 15),
        GeoNextStep("Add an FAQ that answers the top 5–8 questions", GeoPriority.MEDIUM, "why", "Add FAQ.", 25),
        GeoNextStep("Add a TL;DR box (2–4 bullets) near the top", GeoPriority.LOW, "why", "Add TL;DR.", 10),
    ]
    signals = ContentSignals(500, 2, 1, 3, 0, False, False, False, False, True, False, False, True, 0.1, 0.5)
    return GeoNextStepsReport("summary", steps, signals, GeoIntent.INFORMATIONAL)


class TestHeadline:
    def test_quality_and_not_cited(self):
        snap = _snapshot()
        rating = rate_page(snap, FakeClassifier({"page_quality": 3.5}))
        assert build_headline(rating, _access(snap), _probes([False, False, False]), None) == \
            "Google would rate this page High+. AI search engines aren't citing it."

    def test_partial_citations(self):
        snap = _snapshot()
        rating = rate_page(snap, FakeClassifier())
        assert build_headline(rating, _access(snap), _probes([True, False, False]), None).endswith(
            "AI search engines cite it in 1 of 3 answers.")

    def test_blocked_search_bots_beat_probes(self):
        snap = _snapshot()
        access = _access(snap, "User-agent: PerplexityBot\nDisallow: /\n")
        assert "blocks the crawlers" in build_headline(rate_page(snap, FakeClassifier()), access, _probes([True]), None)

    def test_falls_back_to_simscore_without_probes(self):
        snap = _snapshot()
        h = build_headline(rate_page(snap, FakeClassifier()), _access(snap), None, _readiness(59))
        assert h == "Google would rate this page High. Its content isn't shaped for AI answers yet."

    def test_unrated_csr_page(self):
        spa = parse_snapshot("https://app.example/", (FIXTURES / "thin_spa.html").read_text())
        rating = rate_page(spa, FakeClassifier())
        access = build_ai_access_report(spa, SiteFiles("User-agent: *\nAllow: /", 200, False, 404))
        h = build_headline(rating, access, None, None)
        assert h.startswith("SimCheck can't see this page's content.")

    def test_nothing_available(self):
        assert build_headline(None, None, None, None) == "Analysis complete."


class TestBuildReport:
    def test_full_report(self):
        snap = _snapshot()
        rating = rate_page(snap, FakeClassifier({"page_quality": 3.5, "ymyl": "clearly", "ymyl_topic": "health",
                                                 "experience": 2.0}))
        report = build_report(snap, rating, _access(snap), _readiness(59), _geo(), _probes([False, False, False]))
        quality, visibility = report.summary
        assert quality.startswith("The page clears Google's bar for a health topic where accuracy matters.")
        assert "names its author" in quality and "uses Article schema" in quality and "links 2 outside sources" in quality
        assert "updated in August 2026" in quality
        assert "Every AI search crawler can reach it." in visibility
        assert "wasn't cited in any of the 3 AI answers" in visibility and "mayoclinic.org" in visibility
        assert "covers the target query unevenly" in visibility
        assert report.quality_line == "YMYL health page, trust rated High."
        assert report.visibility_line.startswith("Cited instead: ")
        assert report.simscore_line == "Covers the target query at 52/100."
        titles = [f.title for f in report.fixes]
        assert titles[0].startswith("Front-load a direct answer")
        assert "Add a first-hand perspective" in titles
        assert len(report.fixes) == 3

    def test_access_fixes_rank_first(self):
        snap = _snapshot()
        access = _access(snap, "User-agent: OAI-SearchBot\nDisallow: /\n")
        report = build_report(snap, rate_page(snap, FakeClassifier()), access, None, _geo(), None)
        assert report.fixes[0].title == "Allow AI search crawlers in robots.txt"
        assert "OAI-SearchBot" in report.fixes[0].detail

    def test_minimal_inputs(self):
        report = build_report(readiness=_readiness(85))
        assert report.headline == "It is well shaped for AI answers."
        assert report.fixes == ()

    def test_no_em_dashes_in_generated_copy(self):
        snap = _snapshot()
        report = build_report(snap, rate_page(snap, FakeClassifier()), _access(snap), _readiness(59), None,
                              _probes([False]))
        text = " ".join([report.headline, *report.summary, report.quality_line, report.visibility_line])
        assert "—" not in text


def test_rank_fixes_dedupes_and_orders():
    fixes = [Fix("B", "", 25, "content", 4), Fix("A", "", 10, "content", 4), Fix("A", "", 5, "content", 4),
             Fix("Z", "", None, "access", 0)]
    assert [f.title for f in rank_fixes(fixes, 5)] == ["Z", "A", "B"]
