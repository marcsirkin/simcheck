"""
Site audit (Feature 14): sample a site's pages and rate each one.

Discovery reads robots.txt Sitemap: lines, then /sitemap.xml and
/sitemap_index.xml, following sitemap indexes breadth-first. Without a
sitemap it falls back to links on the homepage. Only same-site URLs that
robots.txt allows for SimCheck are kept.

Sitemaps are read newest-first by their <lastmod>: large sites list
hundreds of archive sitemaps oldest-first (martech.org: 318), and reading
them in order audited 2020-era posts. By default the sample is drawn from
pages updated in the last 12 months, falling back to the newest dated
pages, then to everything when the sitemap carries no dates.

Sampling is stratified by first path segment (/blog, /health, ...) so one
large section doesn't crowd out the rest.

Sitemap XML is parsed with a regex over <loc> elements rather than an XML
parser: the input is untrusted, and this avoids entity-expansion attacks
while handling the malformed sitemaps common in the wild.
"""

from __future__ import annotations

import random
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from simcheck.quality.classifier import ClassifierError, QualityClassifier
from simcheck.quality.page_quality import rate_page
from simcheck.quality.snapshot import (
    SnapshotError,
    _same_site,
    fetch_page,
    parse_snapshot,
    validate_url,
)


DEFAULT_SAMPLE = 25
MAX_SAMPLE = 100
MAX_DISCOVERED_URLS = 5000
MAX_SITEMAPS = 20
AUDIT_WORKERS = 4
ROBOTS_AGENT = "SimCheck"

_LOC_RE = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*([^<\]\s]+)\s*(?:\]\]>)?\s*</loc>", re.IGNORECASE)
_SITEMAP_BLOCK_RE = re.compile(r"<sitemap\b.*?</sitemap>", re.IGNORECASE | re.DOTALL)
_URL_BLOCK_RE = re.compile(r"<url\b.*?</url>", re.IGNORECASE | re.DOTALL)
_LASTMOD_RE = re.compile(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", re.IGNORECASE)

DEFAULT_RECENT_DAYS = 365
# When too few pages fall in the recent window, sample from this many of
# the newest dated pages (per requested page) instead.
NEWEST_POOL_FACTOR = 4
_SITEMAP_LINE_RE = re.compile(r"^\s*sitemap\s*:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
# Non-page resources that sometimes appear in sitemaps or homepage links
_SKIP_EXT_RE = re.compile(r"\.(pdf|jpe?g|png|gif|webp|svg|mp4|mp3|zip|xml|css|js)(\?|$)", re.IGNORECASE)


class SiteAuditError(Exception):
    """Raised when a site can't be discovered or audited at all."""


@dataclass(frozen=True)
class SitePage:
    """One audited page (a row in the site grid)."""
    url: str
    path: str
    title: str
    rated: bool
    band: str
    level: Optional[float]
    pq_score: Optional[float]
    ymyl: Optional[str]
    trust: Optional[float]
    author: Optional[str]
    updated: Optional[str]
    words: int
    gates: tuple
    error: Optional[str] = None


@dataclass(frozen=True)
class SiteAudit:
    """Site audit result."""
    site: str
    source: str            # where URLs came from (sitemap URL or "homepage links")
    discovered: int
    pages: tuple
    pool: str = ""         # which pages the sample was drawn from, for display

    @property
    def rated_pages(self) -> list:
        return [p for p in self.pages if p.rated and p.error is None]

    def summary(self) -> dict:
        """Headline numbers for the Site Audit tab."""
        rated = self.rated_pages
        if not rated:
            return {"sampled": len(self.pages), "rated": 0}
        return {
            "sampled": len(self.pages),
            "rated": len(rated),
            "high_or_better": sum(1 for p in rated if p.level >= 3.0),
            "low_or_worse": sum(1 for p in rated if p.level <= 1.0),
            "avg_pq": mean(p.pq_score for p in rated),
            "clearly_ymyl": sum(1 for p in rated if p.ymyl == "clearly"),
            "named_author": sum(1 for p in rated if p.author),
            "unrated": sum(1 for p in self.pages if not p.rated and p.error is None),
            "errors": sum(1 for p in self.pages if p.error),
            "lowest": min(rated, key=lambda p: p.pq_score),
        }


# =============================================================================
# Pure helpers
# =============================================================================

def _entries(blocks: list) -> list:
    """(loc, lastmod date or None) for each <sitemap>/<url> block with a <loc>."""
    out = []
    for block in blocks:
        loc = _LOC_RE.search(block)
        if loc:
            lastmod = _LASTMOD_RE.search(block)
            out.append((loc.group(1), lastmod.group(1)[:10] if lastmod else None))
    return out


def parse_sitemap(xml: str) -> tuple:
    """
    Split a sitemap document into child sitemaps and page URLs.

    Returns:
        (child sitemaps, pages), each a list of (url, lastmod "YYYY-MM-DD" or None)
    """
    children = _entries(_SITEMAP_BLOCK_RE.findall(xml))
    stripped = _SITEMAP_BLOCK_RE.sub("", xml)
    pages = _entries(_URL_BLOCK_RE.findall(stripped))
    if not pages:
        # Malformed sitemaps sometimes list bare <loc> elements
        pages = [(loc, None) for loc in _LOC_RE.findall(stripped)]
    return children, pages


def choose_pool(urls: list, lastmods: dict, sample_size: int,
                recent_days: Optional[int] = DEFAULT_RECENT_DAYS, today: Optional[date] = None) -> tuple:
    """
    Pick which discovered URLs to sample from.

    Args:
        urls: Discovered URLs
        lastmods: {url: "YYYY-MM-DD"} from the sitemaps
        sample_size: Pages wanted
        recent_days: Recency window in days; None = whole site
        today: Override for tests

    Returns:
        (pool of URLs, plain-language description of the pool)
    """
    if not recent_days:
        return list(urls), "the whole site"
    dated = [(u, lastmods[u]) for u in urls if lastmods.get(u)]
    if not dated:
        return list(urls), "the whole site (the sitemap has no dates)"
    cutoff = ((today or date.today()) - timedelta(days=recent_days)).isoformat()
    window = "the last 12 months" if recent_days == 365 else f"the last {recent_days} days"
    recent = [u for u, d in dated if d >= cutoff]
    if len(recent) >= sample_size:
        return recent, f"pages updated in {window}"
    newest = sorted(dated, key=lambda ud: ud[1], reverse=True)[:max(sample_size * NEWEST_POOL_FACTOR, sample_size)]
    return [u for u, _ in newest], f"the newest pages (few updated in {window})"


def sitemaps_from_robots(robots_txt: str) -> list:
    """Sitemap URLs declared in robots.txt."""
    return _SITEMAP_LINE_RE.findall(robots_txt or "")


def site_origin(site: str) -> str:
    """Normalize user input ('example.com', 'https://example.com/x') to an origin."""
    site = site.strip()
    if "//" not in site:
        site = "https://" + site
    parsed = urlparse(site)
    if not parsed.netloc:
        raise SiteAuditError(f"Not a site: {site!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def filter_urls(urls: list, origin: str, robots: Optional[RobotFileParser] = None) -> list:
    """Same-site, page-like, robots-allowed, de-duplicated URLs (order kept)."""
    host = urlparse(origin).netloc
    out = OrderedDict()
    for u in urls:
        u = u.split("#")[0].strip()
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https") or not _same_site(parsed.netloc, host):
            continue
        if _SKIP_EXT_RE.search(parsed.path):
            continue
        if robots is not None and not robots.can_fetch(ROBOTS_AGENT, u):
            continue
        out[u] = None
    return list(out)


def _section(url: str) -> str:
    """First path segment, used as the sampling stratum."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[0] if len(parts) > 1 else "/"


def sample_urls(urls: list, n: int, seed: int = 7) -> list:
    """
    Stratified sample: round-robin across path sections, random within each.

    Deterministic for a given seed so re-runs audit the same pages.
    """
    if n >= len(urls):
        return list(urls)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for u in urls:
        groups[_section(u)].append(u)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    picked = []
    while len(picked) < n:
        for key in order:
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return picked


# =============================================================================
# Network
# =============================================================================

def _fetch_text(url: str, fetcher: Callable) -> Optional[str]:
    try:
        return fetcher(url).html
    except SnapshotError:
        return None


def discover_urls(site: str, fetcher: Callable = fetch_page) -> tuple:
    """
    Find page URLs for a site.

    Args:
        site: Domain or URL
        fetcher: fetch_page-compatible callable (tests inject a fake)

    Returns:
        (origin, source description, filtered URLs, {url: lastmod})

    Raises:
        SiteAuditError: If nothing can be discovered
    """
    origin = site_origin(site)
    try:
        validate_url(origin)
    except SnapshotError as e:
        raise SiteAuditError(str(e)) from e

    robots_txt = _fetch_text(f"{origin}/robots.txt", fetcher)
    robots = None
    if robots_txt is not None:
        robots = RobotFileParser()
        robots.parse(robots_txt.splitlines())

    # Queue of (lastmod, sitemap url). Roots have no date and go first; after
    # that the newest child sitemap is always read next.
    roots = sitemaps_from_robots(robots_txt or "") or [f"{origin}/sitemap.xml", f"{origin}/sitemap_index.xml"]
    queue = [(None, sm) for sm in roots]
    seen, urls, sources, lastmods = set(), [], [], {}
    while queue and len(seen) < MAX_SITEMAPS and len(urls) < MAX_DISCOVERED_URLS:
        _, sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        xml = _fetch_text(sm, fetcher)
        if not xml or "<loc" not in xml.lower():
            continue
        children, pages = parse_sitemap(xml)
        queue += [(lm, c) for c, lm in children if c not in seen]
        queue = [q for q in queue if q[0] is None] + sorted(
            (q for q in queue if q[0] is not None), key=lambda q: q[0], reverse=True)
        kept = filter_urls([loc for loc, _ in pages], origin, robots)
        if kept:
            # Report a sitemap that actually contributed pages, not e.g. a
            # foreign-language subdomain sitemap whose URLs were all filtered.
            sources.append(sm)
            urls += kept
            page_dates = dict(pages)
            for u in kept:
                if page_dates.get(u):
                    lastmods[u] = page_dates[u]

    urls = filter_urls(urls[:MAX_DISCOVERED_URLS], origin, robots)
    if urls:
        return origin, sources[0], urls, lastmods

    home = _fetch_text(origin + "/", fetcher)
    if home:
        links = [urljoin(origin + "/", a["href"]) for a in BeautifulSoup(home, "html.parser").find_all("a", href=True)]
        urls = filter_urls(links, origin, robots)
        if urls:
            return origin, "homepage links", urls, {}
    raise SiteAuditError(f"No sitemap or crawlable links found for {origin}.")


def _audit_one(url: str, classifier: QualityClassifier, fetcher: Callable) -> SitePage:
    path = urlparse(url).path or "/"
    try:
        fetched = fetcher(url)
        snap = parse_snapshot(fetched.url, fetched.html, fetched.final_url, fetched.status_code, fetched.headers)
        rating = rate_page(snap, classifier)
    except (SnapshotError, ClassifierError) as e:
        return SitePage(url, path, "", False, "Error", None, None, None, None, None, None, 0, (), error=str(e))
    return SitePage(
        url=url, path=path, title=snap.title, rated=rating.rated, band=rating.band, level=rating.level,
        pq_score=rating.pq_score, ymyl=rating.ymyl,
        trust=rating.eeat.get("trust") if rating.rated else None,
        author=snap.author_name, updated=((snap.modified or snap.published) or "")[:10] or None,
        words=snap.word_count, gates=rating.gates,
    )


def audit_site(
    site: str,
    classifier: QualityClassifier,
    sample_size: int = DEFAULT_SAMPLE,
    progress: Optional[Callable] = None,
    fetcher: Callable = fetch_page,
    workers: int = AUDIT_WORKERS,
    recent_days: Optional[int] = DEFAULT_RECENT_DAYS,
) -> SiteAudit:
    """
    Discover, sample, and rate a site's pages.

    Per-page failures become error rows; the audit only fails if discovery
    fails.

    Args:
        site: Domain or URL
        classifier: Rating backend (Jev recommended: one cheap call per page)
        sample_size: Pages to rate (capped at MAX_SAMPLE)
        progress: Optional callback(done, total)
        fetcher: fetch_page-compatible callable
        workers: Concurrent page fetches/ratings
        recent_days: Sample pages updated within this many days (None = whole site)

    Returns:
        SiteAudit with pages in sample order

    Raises:
        SiteAuditError: If no URLs can be discovered
    """
    sample_size = max(1, min(sample_size, MAX_SAMPLE))
    origin, source, urls, lastmods = discover_urls(site, fetcher)
    pool, pool_label = choose_pool(urls, lastmods, sample_size, recent_days)
    sample = sample_urls(pool, sample_size)

    pages, done = [None] * len(sample), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_audit_one, u, classifier, fetcher): i for i, u in enumerate(sample)}
        for future in futures:
            pages[futures[future]] = future.result()
            done += 1
            if progress:
                progress(done, len(sample))

    return SiteAudit(site=origin, source=source or "", discovered=len(urls), pages=tuple(pages), pool=pool_label)


def audit_to_rows(audit: SiteAudit) -> list:
    """Flat dict rows for CSV export and the sortable grid."""
    return [{
        "url": p.url, "path": p.path, "title": p.title, "band": p.band,
        "pq_score": None if p.pq_score is None else round(p.pq_score),
        "trust": None if p.trust is None else round(p.trust, 1),
        "ymyl": p.ymyl, "author": p.author, "updated": p.updated, "words": p.words,
        "overrides": "; ".join(p.gates), "error": p.error,
    } for p in audit.pages]
