"""Tests for site discovery, sampling, and audit (simcheck.quality.site). No network."""

from collections import Counter
from urllib.robotparser import RobotFileParser

import pytest

from simcheck.quality import site as site_mod
from simcheck.quality.classifier import FakeClassifier
from simcheck.quality.site import (
    SiteAuditError,
    audit_site,
    audit_to_rows,
    discover_urls,
    filter_urls,
    parse_sitemap,
    sample_urls,
    site_origin,
    sitemaps_from_robots,
)
from simcheck.quality.snapshot import FetchResult, SnapshotError


INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://ex.com/sm-posts.xml</loc></sitemap>
  <sitemap><loc><![CDATA[https://ex.com/sm-pages.xml]]></loc></sitemap>
</sitemapindex>"""
POSTS = """<urlset><url><loc>https://ex.com/blog/a</loc></url><url><loc> https://ex.com/blog/b </loc></url>
<url><loc>https://ex.com/blog/c</loc></url><url><loc>https://other.com/x</loc></url></urlset>"""
PAGES = "<urlset><url><loc>https://www.ex.com/about</loc></url><url><loc>https://ex.com/private/p</loc></url></urlset>"
ARTICLE = "<html><head><title>T</title></head><body><main><p>" + "Real content words here. " * 30 + "</p></main></body></html>"


def _fetcher(pages: dict):
    def fetch(url):
        if url not in pages:
            raise SnapshotError(f"HTTP 404 for {url}")
        return FetchResult(url, url, 200, pages[url], {})
    return fetch


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    monkeypatch.setattr(site_mod, "validate_url", lambda u: u)


class TestParsing:
    def test_parse_index(self):
        children, pages = parse_sitemap(INDEX)
        assert children == ["https://ex.com/sm-posts.xml", "https://ex.com/sm-pages.xml"] and pages == []

    def test_parse_urlset_trims(self):
        _, pages = parse_sitemap(POSTS)
        assert pages[1] == "https://ex.com/blog/b"

    def test_robots_sitemap_lines(self):
        assert sitemaps_from_robots("User-agent: *\nSitemap: https://ex.com/a.xml\nsitemap:https://ex.com/b.xml") == [
            "https://ex.com/a.xml", "https://ex.com/b.xml"]

    @pytest.mark.parametrize("raw,origin", [("ex.com", "https://ex.com"), ("http://ex.com/x/y", "http://ex.com"),
                                            (" https://www.ex.com ", "https://www.ex.com")])
    def test_site_origin(self, raw, origin):
        assert site_origin(raw) == origin


class TestFilterAndSample:
    def test_filter(self):
        robots = RobotFileParser()
        robots.parse(["User-agent: *", "Disallow: /private/"])
        urls = ["https://ex.com/a", "https://ex.com/a#top", "https://www.ex.com/b", "https://other.com/c",
                "https://ex.com/file.pdf", "mailto:x@ex.com", "https://ex.com/private/p"]
        assert filter_urls(urls, "https://ex.com", robots) == ["https://ex.com/a", "https://www.ex.com/b"]

    def test_sample_is_stratified_and_deterministic(self):
        urls = [f"https://ex.com/blog/{i}" for i in range(50)] + [f"https://ex.com/docs/{i}" for i in range(5)]
        s1, s2 = sample_urls(urls, 10), sample_urls(urls, 10)
        assert s1 == s2 and len(set(s1)) == 10
        sections = Counter(u.split("/")[3] for u in s1)
        assert sections["docs"] == 5 and sections["blog"] == 5

    def test_sample_returns_all_when_small(self):
        assert sample_urls(["https://ex.com/a"], 10) == ["https://ex.com/a"]


class TestDiscover:
    def test_follows_robots_sitemap_and_index(self):
        fetch = _fetcher({
            "https://ex.com/robots.txt": "User-agent: *\nDisallow: /private/\nSitemap: https://ex.com/index.xml",
            "https://ex.com/index.xml": INDEX,
            "https://ex.com/sm-posts.xml": POSTS,
            "https://ex.com/sm-pages.xml": PAGES,
        })
        origin, source, urls = discover_urls("ex.com", fetch)
        assert origin == "https://ex.com" and source == "https://ex.com/sm-posts.xml"
        assert urls == ["https://ex.com/blog/a", "https://ex.com/blog/b", "https://ex.com/blog/c", "https://www.ex.com/about"]

    def test_default_sitemap_location(self):
        fetch = _fetcher({"https://ex.com/sitemap.xml": POSTS})
        _, source, urls = discover_urls("https://ex.com", fetch)
        assert source == "https://ex.com/sitemap.xml" and len(urls) == 3

    def test_homepage_fallback(self):
        fetch = _fetcher({"https://ex.com/": '<a href="/one">1</a><a href="https://other.com/">x</a><a href="/two">2</a>'})
        _, source, urls = discover_urls("ex.com", fetch)
        assert source == "homepage links" and urls == ["https://ex.com/one", "https://ex.com/two"]

    def test_nothing_found(self):
        with pytest.raises(SiteAuditError, match="No sitemap"):
            discover_urls("ex.com", _fetcher({}))


class TestAudit:
    def test_audit_rates_and_records_errors(self):
        fetch = _fetcher({
            "https://ex.com/sitemap.xml": POSTS,
            "https://ex.com/blog/a": ARTICLE,
            "https://ex.com/blog/b": "<html><body><div id='root'></div></body></html>",
        })
        progress = []
        audit = audit_site("ex.com", FakeClassifier(), sample_size=10, fetcher=fetch, workers=2,
                           progress=lambda d, t: progress.append((d, t)))
        by_path = {p.path: p for p in audit.pages}
        assert by_path["/blog/a"].rated and by_path["/blog/a"].band == "High"
        assert not by_path["/blog/b"].rated and by_path["/blog/b"].error is None
        assert by_path["/blog/c"].error and "404" in by_path["/blog/c"].error
        assert progress[-1] == (3, 3)
        s = audit.summary()
        assert s["rated"] == 1 and s["unrated"] == 1 and s["errors"] == 1 and s["high_or_better"] == 1
        assert s["lowest"].path == "/blog/a"

    def test_rows_for_export(self):
        fetch = _fetcher({"https://ex.com/sitemap.xml": "<urlset><url><loc>https://ex.com/a/b</loc></url></urlset>",
                          "https://ex.com/a/b": ARTICLE})
        rows = audit_to_rows(audit_site("ex.com", FakeClassifier(), fetcher=fetch))
        assert rows[0]["path"] == "/a/b" and rows[0]["pq_score"] == 75 and rows[0]["band"] == "High"

    def test_sample_size_capped(self):
        urls = "".join(f"<url><loc>https://ex.com/p/{i}</loc></url>" for i in range(150))
        pages = {"https://ex.com/sitemap.xml": f"<urlset>{urls}</urlset>"}
        pages.update({f"https://ex.com/p/{i}": ARTICLE for i in range(150)})
        audit = audit_site("ex.com", FakeClassifier(), sample_size=500, fetcher=_fetcher(pages))
        assert len(audit.pages) == site_mod.MAX_SAMPLE

    def test_summary_with_nothing_rated(self):
        audit = site_mod.SiteAudit("https://ex.com", "", 0, ())
        assert audit.summary() == {"sampled": 0, "rated": 0}
