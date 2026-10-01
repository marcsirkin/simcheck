"""
Page fetching and deterministic extraction (Feature 9).

Turns a URL into a PageSnapshot: metadata, schema, authorship, dates, main
content (text + Markdown), GEO evidence counts, reputation links, and
ad/affiliate signals. No API keys, no LLM calls.

Network and parsing are separated:
- fetch_page(): validated HTTP fetch (scheme check, private-IP block on every
  redirect hop, response size cap)
- parse_snapshot(): pure function over HTML, used directly by tests

Extraction logic is ported from the content-evaluator skill's
fetch_content.py so ratings here and in that skill see the same signals.
"""

from __future__ import annotations

import copy
import io
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown, StreamInfo


USER_AGENT = "Mozilla/5.0 (compatible; SimCheck/2.0; +https://sirkin.com)"
FETCH_TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5

# A semantic container must hold at least this share of body words to be
# treated as the main content; otherwise the whole (de-chromed) body is used.
MAIN_CONTENT_MIN_SHARE = 0.4

# Body markers of WAF / bot-challenge pages served with HTTP 200
_CHALLENGE_RE = re.compile(
    r"_Incapsula_Resource|Request unsuccessful\. Incapsula|cf-browser-verification|"
    # Not "challenge-platform": Cloudflare injects that script into normal pages.
    r"Just a moment\.\.\.|Attention Required! \| Cloudflare|"
    r"Pardon Our Interruption|captcha-delivery\.com|px-captcha|"
    r"<title>\s*Access Denied\s*</title>",
    re.IGNORECASE)
CHALLENGE_MAX_BYTES = 20_000

# Non-visible elements, always stripped
_INVISIBLE_TAGS = ["script", "style", "noscript", "template"]
# Navigation chrome, stripped unless that would leave the page empty
_CHROME_TAGS = ["nav", "footer", "header", "aside", "form"]
# If stripping chrome leaves fewer words than this, the site has put real
# content inside chrome elements (e.g. everything in <header>, or ASP.NET's
# page-wide <form>), so fall back to the visible body.
MIN_DECHROMED_WORDS = 50


class SnapshotError(Exception):
    """Raised when a URL is invalid, unsafe, unreachable, or unparseable."""


class FetchBlockedError(SnapshotError):
    """
    The site refused the fetcher (bot protection / WAF).

    Reported as a finding (AI crawlers may hit the same wall). The caller can
    still rate the page from pasted HTML via parse_snapshot().
    """

    def __init__(self, url: str, status_code: int):
        self.url = url
        self.status_code = status_code
        super().__init__(
            f"HTTP {status_code}: {url} blocks automated fetchers. "
            "Paste the page HTML (view-source) to rate it anyway."
        )


# Status codes that typically mean "bot protection said no", not "page missing"
BLOCKED_STATUS_CODES = (401, 403, 429)


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class FetchResult:
    """Raw HTTP fetch outcome."""
    url: str
    final_url: str
    status_code: int
    html: str
    headers: dict


@dataclass(frozen=True)
class ReputationLinks:
    """
    Links to pages QRG raters use to judge who is responsible for a site
    (QRG 2.5.2-2.5.3). Values are absolute URLs or None if not found.
    """
    about: Optional[str] = None
    contact: Optional[str] = None
    privacy: Optional[str] = None
    terms: Optional[str] = None
    editorial_policy: Optional[str] = None

    def found_count(self) -> int:
        """Number of reputation link types present."""
        return sum(1 for v in (self.about, self.contact, self.privacy,
                               self.terms, self.editorial_policy) if v)


@dataclass(frozen=True)
class AdSignals:
    """Heuristic monetization signals (QRG 2.4.3)."""
    ad_slot_count: int
    affiliate_link_count: int
    sponsored_link_count: int


@dataclass(frozen=True)
class PageSnapshot:
    """
    Everything deterministic SimCheck knows about a page.

    main_markdown feeds the existing CCS / content-signal pipeline;
    main_text feeds the classifier state.
    """
    url: str
    final_url: str
    status_code: int
    title: str
    meta_description: str
    canonical: Optional[str]
    robots_meta: Optional[str]
    x_robots_tag: Optional[str]
    lang: Optional[str]
    author_name: Optional[str]
    author_url: Optional[str]
    published: Optional[str]
    modified: Optional[str]
    schema_types: tuple
    json_ld: tuple
    headings: tuple  # ((level, text), ...)
    h1_count: int
    word_count: int
    main_text: str
    main_markdown: str
    paragraph_count: int
    avg_paragraph_words: float
    long_paragraph_count: int  # > 150 words
    sentence_count: int
    avg_sentence_words: float
    long_sentence_count: int  # > 30 words
    stats_mentions: int
    quotation_count: int
    internal_link_count: int
    external_link_count: int
    external_hosts: tuple
    script_count: int  # executable <script> tags, inline and external (not JSON-LD)
    reputation: ReputationLinks
    ads: AdSignals

    @property
    def host(self) -> str:
        """Hostname of the final (post-redirect) URL."""
        return urlparse(self.final_url).netloc


# =============================================================================
# Network: validated fetch
# =============================================================================

def validate_url(url: str) -> str:
    """
    Check that a URL is http(s) and does not resolve to a private address.

    Blocking private/loopback/link-local targets matters once the site
    crawler follows URLs found in third-party sitemaps.

    Args:
        url: URL to validate

    Returns:
        The URL, stripped

    Raises:
        SnapshotError: If the scheme, host, or resolved address is not allowed
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SnapshotError(f"Only http(s) URLs are supported: {url!r}")
    host = parsed.hostname
    if not host:
        raise SnapshotError(f"URL has no host: {url!r}")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise SnapshotError(f"Could not resolve host {host!r}: {e}") from e

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast or address.is_unspecified):
            raise SnapshotError(f"Refusing to fetch non-public address {address} for host {host!r}")
    return url


def _read_capped(response: requests.Response) -> bytes:
    """Read a streamed response body, refusing anything over MAX_RESPONSE_BYTES."""
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise SnapshotError(f"Response exceeds {MAX_RESPONSE_BYTES // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def is_challenge_page(html: str) -> bool:
    """
    True if HTML looks like a WAF/bot challenge served with a 2xx status.

    Only small documents are checked: challenge pages are tiny, and a real
    article that merely mentions "captcha" must not be misclassified.
    """
    return len(html) <= CHALLENGE_MAX_BYTES and bool(_CHALLENGE_RE.search(html))


_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)


def decode_body(body: bytes, content_type: str) -> str:
    """
    Decode a response body we already read.

    Never use response.apparent_encoding here: the body was consumed by the
    capped stream read, and requests raises "content already consumed".
    Order: declared charset, then UTF-8, then Windows-1252 (which maps
    every byte, so decoding always succeeds).
    """
    match = _CHARSET_RE.search(content_type or "")
    if match:
        try:
            return body.decode(match.group(1), errors="replace")
        except LookupError:
            pass  # unknown charset name; fall through to sniffing
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")


def fetch_page(url: str, timeout: int = FETCH_TIMEOUT_SECONDS) -> FetchResult:
    """
    Fetch a URL, validating every redirect hop.

    Redirects are followed manually so a public URL cannot bounce the
    fetcher onto an internal address.

    Args:
        url: Page URL (http/https)
        timeout: Per-request timeout in seconds

    Returns:
        FetchResult with decoded HTML

    Raises:
        SnapshotError: On invalid/unsafe URL, network error, too many
            redirects, oversized body, or HTTP error status
    """
    original = url
    current = validate_url(url)

    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = requests.get(
                current,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as e:
            raise SnapshotError(f"Fetch failed for {current}: {e}") from e

        if response.is_redirect:
            location = response.headers.get("Location", "")
            response.close()
            current = validate_url(urljoin(current, location))
            continue

        try:
            body = _read_capped(response)
        finally:
            response.close()

        if response.status_code in BLOCKED_STATUS_CODES or (
                response.status_code == 503 and "cf-mitigated" in response.headers):
            raise FetchBlockedError(current, response.status_code)
        if response.status_code >= 400:
            raise SnapshotError(f"HTTP {response.status_code} for {current}")

        html = decode_body(body, response.headers.get("Content-Type", ""))
        if is_challenge_page(html):
            raise FetchBlockedError(current, response.status_code)
        return FetchResult(
            url=original,
            final_url=current,
            status_code=response.status_code,
            html=html,
            headers=dict(response.headers),
        )

    raise SnapshotError(f"Too many redirects (>{MAX_REDIRECTS}) for {original}")


# =============================================================================
# Parsing helpers (ported from content-evaluator fetch_content.py)
# =============================================================================

def _extract_json_ld(soup: BeautifulSoup) -> list:
    """Parsed JSON-LD blocks; malformed blocks are skipped (common in the wild)."""
    blocks = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            blocks.append(json.loads(tag.string or ""))
        except (json.JSONDecodeError, TypeError):
            continue
    return blocks


def _walk_schema(obj):
    """Yield every dict nested anywhere in a JSON-LD structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_schema(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_schema(v)


def _schema_types(blocks: list) -> list:
    """Flat, sorted, de-duplicated list of @type values."""
    types = set()
    for node in _walk_schema(blocks):
        t = node.get("@type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            types.update(x for x in t if isinstance(x, str))
    return sorted(types)


def _find_in_schema(blocks: list, key: str):
    """First non-empty value for key anywhere in the schema blocks."""
    for node in _walk_schema(blocks):
        if node.get(key):
            return node[key]
    return None


_JS_SCRIPT_TYPES = ("", "text/javascript", "application/javascript", "module")


def _is_executable_script(tag) -> bool:
    """True for JavaScript <script> tags; False for data blocks like JSON-LD."""
    return (tag.get("type") or "").strip().lower() in _JS_SCRIPT_TYPES


def _word_count(el) -> int:
    """Words of visible text in an element."""
    return len(el.get_text(" ", strip=True).split())


def _largest(elements) -> Optional[object]:
    """Element with the most text, or None."""
    return max(elements, key=_word_count, default=None)


def _main_content(soup: BeautifulSoup):
    """
    Best-effort main content element. MUTATES soup (removes chrome).

    Semantic containers win only if they hold a real share of the page's
    text. Homepages often use <article> or *content* classes for small cards
    (a metric tile, a testimonial); taking the first match would rate a
    20-word card instead of the page.
    """
    for tag in soup(_INVISIBLE_TAGS):
        tag.decompose()
    visible_body = copy.copy(soup.find("body") or soup)
    for tag in soup(_CHROME_TAGS):
        tag.decompose()
    body = soup.find("body") or soup
    total = _word_count(body)
    if total < MIN_DECHROMED_WORDS and _word_count(visible_body) > total:
        return visible_body
    if total == 0:
        return body

    content_class = lambda c: c and any(k in c.lower() for k in ["content", "article", "post", "entry"])
    candidates = [
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        _largest(soup.find_all("article")),
        _largest(soup.find_all("div", class_=content_class)),
    ]
    for c in candidates:
        if c is not None and _word_count(c) >= MAIN_CONTENT_MIN_SHARE * total:
            return c
    return body


def _count_statistics(text: str) -> int:
    """Statistic-like mentions (%, $, 3x, 5-fold, 10,000) in one pass to avoid double counts."""
    combined = (
        r"\$\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|trillion|M|B|K))?"
        r"|\b\d+(?:\.\d+)?%"
        r"|\b\d+(?:\.\d+)?x\b"
        r"|\b\d+(?:\.\d+)?-fold\b"
        r"|\b\d{2,}(?:,\d{3})+\b"
    )
    return len(re.findall(combined, text, flags=re.IGNORECASE))


def _count_quotations(main_el) -> int:
    """Blockquotes plus 50+ char quoted passages outside blockquotes."""
    blockquotes = main_el.find_all("blockquote")
    text = main_el.get_text(" ", strip=True)
    for bq in blockquotes:
        bq_text = bq.get_text(" ", strip=True)
        if bq_text:
            text = text.replace(bq_text, " ")
    long_quotes = len(re.findall(r"[“\"][^”\"]{50,}[”\"]", text))
    return len(blockquotes) + long_quotes


def _split_sentences(text: str) -> list:
    """Simple splitter; good enough for length statistics."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def _same_site(host: str, base_host: str) -> bool:
    """True if host is base_host or differs only by a leading www."""
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return strip(host) == strip(base_host)


# Link text / href patterns for QRG "who is responsible" pages
_REPUTATION_PATTERNS = {
    "about": re.compile(r"^about(\s+us)?\b|\babout us\b|/about(-us)?(/|$)", re.IGNORECASE),
    "contact": re.compile(r"\bcontact(\s+us)?\b|/contact", re.IGNORECASE),
    "privacy": re.compile(r"\bprivacy\b", re.IGNORECASE),
    "terms": re.compile(r"\bterms\b|terms-of|/tos\b", re.IGNORECASE),
    "editorial_policy": re.compile(
        r"editorial|fact.?check|corrections policy|ethics policy|how we (test|review)",
        re.IGNORECASE),
}
_AUTHOR_HREF_RE = re.compile(r"/(author|authors|people|team|staff|contributors?|bio)/", re.IGNORECASE)
_AFFILIATE_RE = re.compile(
    r"amzn\.to|amazon\.[a-z.]+/.*[?&]tag=|[?&](aff|affiliate|ref|aff_id)=|/go/|/recommends/"
    r"|shareasale|awin1|impact\.com|cj\.com|skimlinks|rakuten",
    re.IGNORECASE)
_AD_CLASS_RE = re.compile(r"(^|[\s_-])(ad|ads|advert|advertisement|adslot|sponsored|dfp|gpt-ad)([\s_-]|$)",
                          re.IGNORECASE)
_AD_IFRAME_RE = re.compile(r"doubleclick|googlesyndication|adservice|taboola|outbrain", re.IGNORECASE)


def _reputation_links(full_soup: BeautifulSoup, base_url: str) -> ReputationLinks:
    """Find About/Contact/Privacy/Terms/Editorial links anywhere on the page (incl. nav/footer)."""
    found: dict = {}
    base_host = urlparse(base_url).netloc
    for a in full_soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        # Only the site's own pages say who is responsible for it; an
        # outbound "facts-about-..." link must not count as an About page.
        if not _same_site(urlparse(absolute).netloc, base_host):
            continue
        haystack = f"{a.get_text(' ', strip=True)} {urlparse(absolute).path}"
        for kind, pattern in _REPUTATION_PATTERNS.items():
            if kind not in found and pattern.search(haystack):
                found[kind] = absolute
    return ReputationLinks(**found)


def _author_url(full_soup: BeautifulSoup, json_ld: list, base_url: str) -> Optional[str]:
    """Author profile URL from schema author.url, rel=author, or author-like hrefs."""
    author = _find_in_schema(json_ld, "author")
    if isinstance(author, list) and author:
        author = author[0]
    if isinstance(author, dict) and isinstance(author.get("url"), str):
        return urljoin(base_url, author["url"])
    rel_author = full_soup.find(["a", "link"], rel="author", href=True)
    if rel_author:
        return urljoin(base_url, rel_author["href"])
    for a in full_soup.find_all("a", href=True):
        if _AUTHOR_HREF_RE.search(a["href"]):
            return urljoin(base_url, a["href"])
    return None


def _author_name(full_soup: BeautifulSoup, json_ld: list) -> Optional[str]:
    """Author name from schema, falling back to meta[name=author]."""
    author = _find_in_schema(json_ld, "author")
    if isinstance(author, list) and author:
        author = author[0]
    if isinstance(author, dict):
        name = author.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    elif isinstance(author, str) and author.strip():
        return author.strip()
    meta = full_soup.find("meta", attrs={"name": "author"})
    if meta and meta.get("content", "").strip():
        return meta["content"].strip()
    return None


def _dates(full_soup: BeautifulSoup, json_ld: list) -> tuple:
    """(published, modified) from JSON-LD, then OpenGraph, then <time>."""
    published = _find_in_schema(json_ld, "datePublished")
    modified = _find_in_schema(json_ld, "dateModified")
    if not published:
        og = full_soup.find("meta", property="article:published_time")
        published = og.get("content") if og else None
    if not modified:
        og = full_soup.find("meta", property="article:modified_time")
        modified = og.get("content") if og else None
    if not published:
        time_tag = full_soup.find("time", datetime=True)
        published = time_tag.get("datetime") if time_tag else None
    return (str(published) if published else None, str(modified) if modified else None)


def _ad_signals(full_soup: BeautifulSoup) -> AdSignals:
    """Count ad slots, affiliate links, and rel=sponsored links."""
    ad_slots = 0
    for el in full_soup.find_all(["div", "aside", "ins", "section"]):
        attrs = " ".join(el.get("class", [])) + " " + (el.get("id") or "")
        if _AD_CLASS_RE.search(attrs) or el.name == "ins" and "adsbygoogle" in attrs:
            ad_slots += 1
    for frame in full_soup.find_all("iframe", src=True):
        if _AD_IFRAME_RE.search(frame["src"]):
            ad_slots += 1

    affiliate = 0
    sponsored = 0
    for a in full_soup.find_all("a", href=True):
        rel = " ".join(a.get("rel", [])).lower()
        if "sponsored" in rel:
            sponsored += 1
        if _AFFILIATE_RE.search(a["href"]):
            affiliate += 1
    return AdSignals(ad_slot_count=ad_slots, affiliate_link_count=affiliate,
                     sponsored_link_count=sponsored)


def html_to_markdown(html: str) -> str:
    """
    Convert an HTML string to Markdown with markitdown (no network).

    Raises:
        SnapshotError: If conversion fails
    """
    try:
        result = MarkItDown().convert_stream(
            io.BytesIO(html.encode("utf-8")),
            stream_info=StreamInfo(extension=".html", charset="utf-8"),
        )
    except Exception as e:  # markitdown raises several converter-specific types
        raise SnapshotError(f"Markdown conversion failed: {e}") from e
    return (result.text_content or "").strip()


# =============================================================================
# Public API
# =============================================================================

def parse_snapshot(
    url: str,
    html: str,
    final_url: Optional[str] = None,
    status_code: int = 200,
    headers: Optional[dict] = None,
) -> PageSnapshot:
    """
    Extract a PageSnapshot from HTML. Pure: no network access.

    Args:
        url: Requested URL
        html: Page HTML
        final_url: URL after redirects (defaults to url)
        status_code: HTTP status
        headers: Response headers (for X-Robots-Tag)

    Returns:
        PageSnapshot
    """
    final_url = final_url or url
    headers = {k.lower(): v for k, v in (headers or {}).items()}

    # Two parses: full_soup stays intact for head/meta/nav-level signals;
    # _main_content() mutates its own tree when stripping chrome.
    full_soup = BeautifulSoup(html, "html.parser")
    main_el = _main_content(BeautifulSoup(html, "html.parser"))

    def _meta(name: str) -> Optional[str]:
        tag = full_soup.find("meta", attrs={"name": name})
        return tag.get("content", "").strip() if tag else None

    title_tag = full_soup.find("title")
    canonical_tag = full_soup.find("link", rel="canonical")
    html_tag = full_soup.find("html")

    json_ld = _extract_json_ld(full_soup)
    published, modified = _dates(full_soup, json_ld)

    main_text = re.sub(r"\s+", " ", main_el.get_text(" ", strip=True)).strip()

    paragraphs = [len(p.get_text(" ", strip=True).split())
                  for p in main_el.find_all("p") if p.get_text(strip=True)]
    sentences = [len(s.split()) for s in _split_sentences(main_text)]

    base_host = urlparse(final_url).netloc
    internal = 0
    external_hosts = set()
    external = 0
    for a in main_el.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        host = urlparse(urljoin(final_url, href)).netloc
        if not host or _same_site(host, base_host):
            internal += 1
        else:
            external += 1
            external_hosts.add(host)

    return PageSnapshot(
        url=url,
        final_url=final_url,
        status_code=status_code,
        title=title_tag.get_text(strip=True) if title_tag else "",
        meta_description=_meta("description") or "",
        canonical=canonical_tag.get("href") if canonical_tag else None,
        robots_meta=_meta("robots"),
        x_robots_tag=headers.get("x-robots-tag"),
        lang=html_tag.get("lang") if html_tag else None,
        author_name=_author_name(full_soup, json_ld),
        author_url=_author_url(full_soup, json_ld, final_url),
        published=published,
        modified=modified,
        schema_types=tuple(_schema_types(json_ld)),
        json_ld=tuple(json_ld),
        headings=tuple((h.name, h.get_text(strip=True))
                       for h in main_el.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
        h1_count=len(main_el.find_all("h1")),
        word_count=len(main_text.split()),
        main_text=main_text,
        main_markdown=html_to_markdown(str(main_el)) if main_text else "",
        paragraph_count=len(paragraphs),
        avg_paragraph_words=round(sum(paragraphs) / len(paragraphs), 1) if paragraphs else 0.0,
        long_paragraph_count=sum(1 for w in paragraphs if w > 150),
        sentence_count=len(sentences),
        avg_sentence_words=round(sum(sentences) / len(sentences), 1) if sentences else 0.0,
        long_sentence_count=sum(1 for w in sentences if w > 30),
        stats_mentions=_count_statistics(main_text),
        quotation_count=_count_quotations(main_el),
        internal_link_count=internal,
        external_link_count=external,
        external_hosts=tuple(sorted(external_hosts)),
        script_count=sum(1 for t in full_soup.find_all("script") if _is_executable_script(t)),
        reputation=_reputation_links(full_soup, final_url),
        ads=_ad_signals(full_soup),
    )


def snapshot_url(url: str) -> PageSnapshot:
    """
    Fetch a URL and extract its PageSnapshot.

    Raises:
        SnapshotError: On fetch or parse failure
    """
    fetched = fetch_page(url)
    return parse_snapshot(
        url=fetched.url,
        html=fetched.html,
        final_url=fetched.final_url,
        status_code=fetched.status_code,
        headers=fetched.headers,
    )


def fetch_markdown(url: str) -> str:
    """
    Fetch a URL and convert the whole page to Markdown.

    Used by the Content Match tab's URL fetcher (replaces markitdown's own
    unvalidated fetch). Converts the full page, matching prior behavior.

    Raises:
        SnapshotError: On fetch failure or empty conversion
    """
    fetched = fetch_page(url)
    markdown = html_to_markdown(fetched.html)
    if not markdown:
        raise SnapshotError("Conversion returned empty content")
    return markdown
