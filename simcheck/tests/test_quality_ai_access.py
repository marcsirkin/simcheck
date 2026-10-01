"""Tests for AI access signals (simcheck.quality.ai_access). No network."""

from pathlib import Path

import pytest

from simcheck.quality.ai_access import (
    AI_BOTS,
    SiteFiles,
    build_ai_access_report,
    evaluate_robots,
)
from simcheck.quality.snapshot import parse_snapshot


FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.com/guides/dkim"

OPEN_ROBOTS = "User-agent: *\nAllow: /\n"
MIXED_ROBOTS = """
User-agent: GPTBot
Disallow: /

User-agent: PerplexityBot
Disallow: /guides/

User-agent: CCBot
Disallow: /

User-agent: *
Allow: /
"""


def _snapshot(name: str, url: str = URL):
    return parse_snapshot(url, (FIXTURES / name).read_text())


def _files(robots=OPEN_ROBOTS, llms=True):
    return SiteFiles(robots_txt=robots, robots_status=200 if robots else 404,
                     llms_txt_present=llms, llms_txt_status=200 if llms else 404)


class TestEvaluateRobots:
    def test_open_allows_all(self):
        assert all(evaluate_robots(OPEN_ROBOTS, URL).values())

    def test_missing_robots_is_unknown(self):
        result = evaluate_robots(None, URL)
        assert set(result) == set(AI_BOTS)
        assert all(v is None for v in result.values())

    def test_per_bot_and_path_rules(self):
        result = evaluate_robots(MIXED_ROBOTS, URL)
        assert result["GPTBot"] is False
        assert result["PerplexityBot"] is False
        assert result["CCBot"] is False
        assert result["ClaudeBot"] is True
        assert evaluate_robots(MIXED_ROBOTS, "https://example.com/blog/x")["PerplexityBot"] is True


class TestReport:
    def test_clean_page_has_only_low_issues(self):
        report = build_ai_access_report(_snapshot("good_article.html"), _files(), "what is DKIM")
        assert report.blocked_bots == ()
        assert not report.noindex and not report.nosnippet
        assert not report.client_rendered_suspect
        assert report.has_article_schema and report.has_organization_schema and report.has_person_schema
        assert all(i.severity == "low" for i in report.issues)

    def test_reuses_core_content_signals(self):
        report = build_ai_access_report(_snapshot("good_article.html"), _files(), "what is DKIM")
        assert report.content_signals.h2_count == 1
        assert report.content_signals.h3_count == 1

    def test_search_vs_training_bots_split(self):
        report = build_ai_access_report(_snapshot("good_article.html"), _files(MIXED_ROBOTS))
        assert set(report.blocked_bots) == {"GPTBot", "PerplexityBot", "CCBot"}
        assert report.search_bots_blocked == ("PerplexityBot",)
        high = [i.message for i in report.issues if i.severity == "high"]
        low = [i.message for i in report.issues if i.severity == "low"]
        assert any("PerplexityBot" in m for m in high)
        assert any("GPTBot" in m and "CCBot" in m for m in low)

    def test_spa_flags_csr_noindex_nosnippet(self):
        report = build_ai_access_report(_snapshot("thin_spa.html", "https://app.example/"),
                                        _files(robots=None, llms=False))
        assert report.noindex and report.nosnippet and report.client_rendered_suspect
        severities = [i.severity for i in report.issues]
        assert severities == sorted(severities, key={"high": 0, "medium": 1, "low": 2}.get)
        assert severities.count("high") == 3
        assert any("robots.txt missing" in i.message for i in report.issues)

    @pytest.mark.parametrize("header,expected", [
        ("noindex", (True, False)),
        ("max-snippet: 0", (False, True)),
        ("none", (True, False)),
        ("index, follow", (False, False)),
    ])
    def test_x_robots_tag_directives(self, header, expected):
        snapshot = parse_snapshot(URL, (FIXTURES / "good_article.html").read_text(),
                                  headers={"X-Robots-Tag": header})
        snapshot = snapshot.__class__(**{**snapshot.__dict__, "robots_meta": None})
        report = build_ai_access_report(snapshot, _files())
        assert (report.noindex, report.nosnippet) == expected

    def test_missing_schema_is_medium(self):
        report = build_ai_access_report(_snapshot("affiliate_page.html"), _files())
        assert any(i.severity == "medium" and "No JSON-LD" in i.message for i in report.issues)


class TestClientRenderedDetection:
    def test_shell_with_single_inline_script_flagged(self):
        html = ('<html><head><title>AI: Voice or Victim</title>'
                '<script type="module">import("/app.js")</script></head>'
                '<body><div id="root"></div></body></html>')
        report = build_ai_access_report(parse_snapshot("https://v.example/", html), _files())
        assert report.client_rendered_suspect

    def test_rich_page_not_flagged(self):
        report = build_ai_access_report(_snapshot("good_article.html"), _files())
        assert not report.client_rendered_suspect
