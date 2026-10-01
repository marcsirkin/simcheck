"""Tests for page fetching and extraction (simcheck.quality.snapshot). No network."""

import socket
from pathlib import Path

import pytest

from simcheck.quality import snapshot as snap
from simcheck.quality.snapshot import (
    MAX_RESPONSE_BYTES,
    FetchBlockedError,
    SnapshotError,
    fetch_page,
    parse_snapshot,
    validate_url,
)


FIXTURES = Path(__file__).parent / "fixtures"
GOOD_URL = "https://example.com/guides/dkim"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def good():
    return parse_snapshot(GOOD_URL, _fixture("good_article.html"))


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address."""
    monkeypatch.setattr(
        snap.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )


class TestParseMetadata:
    def test_head_fields(self, good):
        assert good.title == "What Is DKIM? Email Authentication Explained"
        assert good.meta_description.startswith("DKIM signs")
        assert good.canonical == "https://example.com/guides/dkim"
        assert good.robots_meta == "index, follow"
        assert good.lang == "en"

    def test_schema_skips_malformed_block(self, good):
        assert len(good.json_ld) == 1
        assert good.schema_types == ("Article", "Organization", "Person")

    def test_author_and_dates(self, good):
        assert good.author_name == "Jane Rivera"
        assert good.author_url == "https://example.com/authors/jane-rivera"
        assert good.published == "2026-03-15"
        assert good.modified == "2026-08-01T10:00:00Z"

    def test_x_robots_tag_header(self):
        s = parse_snapshot(GOOD_URL, "<html><body><p>x</p></body></html>",
                           headers={"X-Robots-Tag": "noindex"})
        assert s.x_robots_tag == "noindex"

    def test_host_uses_final_url(self):
        s = parse_snapshot("http://a.com/", "<p>x</p>", final_url="https://www.b.com/x")
        assert s.host == "www.b.com"


class TestParseMainContent:
    def test_nav_and_footer_excluded(self, good):
        assert "Privacy Policy" not in good.main_text
        assert "About us" not in good.main_text
        assert good.main_text.startswith("What Is DKIM?")

    def test_headings(self, good):
        assert good.headings == (
            ("h1", "What Is DKIM?"), ("h2", "How DKIM works"), ("h3", "Key rotation"))
        assert good.h1_count == 1

    def test_markdown_has_structure(self, good):
        assert "## How DKIM works" in good.main_markdown
        assert "### Key rotation" in good.main_markdown

    def test_geo_evidence_counts(self, good):
        assert good.stats_mentions == 2  # 87%, 10,000
        assert good.quotation_count == 1
        assert good.external_link_count == 2
        assert good.internal_link_count == 1
        assert good.external_hosts == ("datatracker.ietf.org", "www.m3aawg.org")

    def test_www_counts_as_internal(self):
        html = '<main><p><a href="https://www.example.com/x">x</a></p></main>'
        s = parse_snapshot("https://example.com/", html)
        assert s.internal_link_count == 1
        assert s.external_link_count == 0

    def test_paragraph_stats(self, good):
        assert good.paragraph_count == 3
        assert good.long_paragraph_count == 0


class TestReputationAndAds:
    def test_reputation_links_found_in_chrome(self, good):
        rep = good.reputation
        assert rep.about == "https://example.com/about-us"
        assert rep.contact == "https://example.com/contact"
        assert rep.privacy == "https://example.com/privacy"
        assert rep.terms == "https://example.com/terms"
        assert rep.editorial_policy == "https://example.com/editorial-standards"
        assert rep.found_count() == 5

    def test_no_reputation_links(self):
        s = parse_snapshot(GOOD_URL, _fixture("affiliate_page.html"))
        assert s.reputation.found_count() == 0

    def test_ad_and_affiliate_signals(self):
        s = parse_snapshot("https://fryers.example/best", _fixture("affiliate_page.html"))
        assert s.ads.ad_slot_count == 4  # div.ad-slot, ins.adsbygoogle, doubleclick iframe, div.sidebar-ads
        assert s.ads.affiliate_link_count == 3
        assert s.ads.sponsored_link_count == 1

    def test_clean_page_has_no_ads(self, good):
        assert good.ads.ad_slot_count == 0
        assert good.ads.affiliate_link_count == 0


class TestThinPage:
    def test_spa_shell(self):
        s = parse_snapshot("https://app.example/", _fixture("thin_spa.html"))
        assert s.word_count == 1
        assert s.script_count == 5
        assert s.robots_meta == "noindex, nosnippet"


class TestValidateUrl:
    @pytest.mark.parametrize("url", ["ftp://x.com/a", "file:///etc/passwd", "javascript:alert(1)", "", "x.com"])
    def test_rejects_non_http(self, url):
        with pytest.raises(SnapshotError):
            validate_url(url)

    @pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"])
    def test_rejects_private_addresses(self, monkeypatch, ip):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        monkeypatch.setattr(snap.socket, "getaddrinfo",
                            lambda host, port: [(family, socket.SOCK_STREAM, 6, "", (ip, port))])
        with pytest.raises(SnapshotError, match="non-public"):
            validate_url("https://internal.example/")

    def test_unresolvable_host(self, monkeypatch):
        def fail(host, port):
            raise socket.gaierror("nope")
        monkeypatch.setattr(snap.socket, "getaddrinfo", fail)
        with pytest.raises(SnapshotError, match="resolve"):
            validate_url("https://nope.invalid/")

    def test_accepts_public(self, public_dns):
        assert validate_url("  https://example.com/a  ") == "https://example.com/a"


class _FakeResponse:
    def __init__(self, status=200, headers=None, body=b"<html><body><p>hi</p></body></html>"):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class TestFetchPage:
    def test_follows_safe_redirect(self, monkeypatch, public_dns):
        responses = iter([
            _FakeResponse(301, {"Location": "/final"}),
            _FakeResponse(200, {"Content-Type": "text/html"}),
        ])
        monkeypatch.setattr(snap.requests, "get", lambda *a, **k: next(responses))
        result = fetch_page("https://example.com/start")
        assert result.final_url == "https://example.com/final"
        assert result.url == "https://example.com/start"
        assert "hi" in result.html

    def test_blocks_redirect_to_private_address(self, monkeypatch):
        def resolve(host, port):
            ip = "169.254.169.254" if host == "metadata.internal" else "93.184.216.34"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
        monkeypatch.setattr(snap.socket, "getaddrinfo", resolve)
        monkeypatch.setattr(snap.requests, "get",
                            lambda *a, **k: _FakeResponse(302, {"Location": "http://metadata.internal/"}))
        with pytest.raises(SnapshotError, match="non-public"):
            fetch_page("https://example.com/")

    def test_redirect_loop_capped(self, monkeypatch, public_dns):
        monkeypatch.setattr(snap.requests, "get",
                            lambda *a, **k: _FakeResponse(302, {"Location": "/loop"}))
        with pytest.raises(SnapshotError, match="Too many redirects"):
            fetch_page("https://example.com/")

    def test_size_cap(self, monkeypatch, public_dns):
        monkeypatch.setattr(snap.requests, "get",
                            lambda *a, **k: _FakeResponse(body=b"x" * (MAX_RESPONSE_BYTES + 1)))
        with pytest.raises(SnapshotError, match="limit"):
            fetch_page("https://example.com/")

    def test_http_error(self, monkeypatch, public_dns):
        monkeypatch.setattr(snap.requests, "get", lambda *a, **k: _FakeResponse(404))
        with pytest.raises(SnapshotError, match="HTTP 404"):
            fetch_page("https://example.com/missing")

    def test_network_error_wrapped(self, monkeypatch, public_dns):
        def boom(*a, **k):
            raise snap.requests.ConnectionError("refused")
        monkeypatch.setattr(snap.requests, "get", boom)
        with pytest.raises(SnapshotError, match="Fetch failed"):
            fetch_page("https://example.com/")

    @pytest.mark.parametrize("status,headers", [(403, {}), (401, {}), (429, {}), (503, {"cf-mitigated": "challenge"})])
    def test_bot_protection_raises_blocked(self, monkeypatch, public_dns, status, headers):
        monkeypatch.setattr(snap.requests, "get", lambda *a, **k: _FakeResponse(status, headers))
        with pytest.raises(FetchBlockedError, match="Paste the page HTML") as exc:
            fetch_page("https://example.com/")
        assert exc.value.status_code == status

    def test_plain_503_is_not_blocked(self, monkeypatch, public_dns):
        monkeypatch.setattr(snap.requests, "get", lambda *a, **k: _FakeResponse(503))
        with pytest.raises(SnapshotError) as exc:
            fetch_page("https://example.com/")
        assert not isinstance(exc.value, FetchBlockedError)
