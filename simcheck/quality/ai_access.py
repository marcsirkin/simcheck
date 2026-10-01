"""
AI access signals: can LLM crawlers and AI search reach and quote this page?
(Feature 9, deterministic half of LLM visibility.)

Checks robots.txt rules per AI user agent, llms.txt presence, indexing and
snippet directives, schema coverage, and client-side-rendering risk. Content
structure/evidence signals are reused from simcheck.core.geo rather than
recomputed.

Network (fetch_site_files) is separate from evaluation
(build_ai_access_report), which is pure and tested with fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from simcheck.core.geo import ContentSignals, extract_content_signals
from simcheck.quality.snapshot import (
    USER_AGENT,
    PageSnapshot,
    SnapshotError,
    validate_url,
)


# AI user agents and what blocking them affects. Google-Extended does NOT
# control AI Overviews / AI Mode (those use Googlebot); it governs Gemini
# training and grounding.
AI_BOTS = {
    "GPTBot": "OpenAI model training",
    "OAI-SearchBot": "ChatGPT search results",
    "ChatGPT-User": "ChatGPT live browsing on user request",
    "ClaudeBot": "Anthropic model training",
    "Claude-SearchBot": "Claude search results",
    "Claude-User": "Claude live fetching on user request",
    "PerplexityBot": "Perplexity search index",
    "Google-Extended": "Gemini training/grounding (not AI Overviews)",
    "CCBot": "Common Crawl (feeds many model training sets)",
}

# Bots whose blocking directly removes the page from an AI answer surface
SEARCH_CRITICAL_BOTS = ("OAI-SearchBot", "ChatGPT-User", "Claude-SearchBot",
                        "Claude-User", "PerplexityBot")

# Below this many main-content words, a page that ships any script is likely
# rendered client-side, so crawlers that don't execute JS see an empty page.
# (Script count alone is a poor signal: Framer/React shells can ship 1-2.)
CSR_WORD_THRESHOLD = 100
CSR_SCRIPT_THRESHOLD = 1

SITE_FILE_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class SiteFiles:
    """Raw site-level files. robots_txt is None when absent or unreachable."""
    robots_txt: Optional[str]
    robots_status: Optional[int]
    llms_txt_present: bool
    llms_txt_status: Optional[int]


@dataclass(frozen=True)
class AccessIssue:
    """One AI-access problem, ordered by severity in AIAccessReport.issues."""
    severity: str  # "high" | "medium" | "low"
    message: str


@dataclass(frozen=True)
class AIAccessReport:
    """
    Deterministic LLM-visibility signals for one page.

    bot_access maps bot name -> True (allowed), False (blocked), or None
    (no robots.txt available, so access is unknown but defaults to allowed).
    """
    bot_access: dict
    blocked_bots: tuple
    llms_txt_present: bool
    noindex: bool
    nosnippet: bool
    schema_types: tuple
    has_article_schema: bool
    has_organization_schema: bool
    has_person_schema: bool
    has_faq_schema: bool
    client_rendered_suspect: bool
    content_signals: ContentSignals
    issues: tuple = field(default_factory=tuple)

    @property
    def search_bots_blocked(self) -> tuple:
        """Blocked bots that directly power AI answer surfaces."""
        return tuple(b for b in self.blocked_bots if b in SEARCH_CRITICAL_BOTS)


def fetch_site_files(url: str) -> SiteFiles:
    """
    Fetch robots.txt and probe llms.txt at the URL's origin.

    Missing files are normal and not errors. Unsafe URLs are.

    Raises:
        SnapshotError: If the URL fails validation
    """
    validate_url(url)
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {"User-Agent": USER_AGENT}

    robots_txt, robots_status = None, None
    try:
        r = requests.get(f"{origin}/robots.txt", headers=headers,
                         timeout=SITE_FILE_TIMEOUT_SECONDS)
        robots_status = r.status_code
        if r.status_code == 200:
            robots_txt = r.text
    except requests.RequestException:
        pass  # unreachable robots.txt == unknown access; reported as None

    llms_present, llms_status = False, None
    try:
        r = requests.get(f"{origin}/llms.txt", headers=headers,
                         timeout=SITE_FILE_TIMEOUT_SECONDS, stream=True)
        llms_status = r.status_code
        # Soft-404s often serve HTML with 200; real llms.txt is plain/markdown
        content_type = r.headers.get("Content-Type", "").lower()
        llms_present = r.status_code == 200 and "html" not in content_type
        r.close()
    except requests.RequestException:
        pass

    return SiteFiles(robots_txt=robots_txt, robots_status=robots_status,
                     llms_txt_present=llms_present, llms_txt_status=llms_status)


def evaluate_robots(robots_txt: Optional[str], url: str) -> dict:
    """
    Per-bot allow/deny for a URL under the given robots.txt.

    Args:
        robots_txt: robots.txt body, or None if unavailable
        url: Page URL to test

    Returns:
        {bot_name: True | False | None}
    """
    if robots_txt is None:
        return {bot: None for bot in AI_BOTS}
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())
    return {bot: parser.can_fetch(bot, url) for bot in AI_BOTS}


def _directives(snapshot: PageSnapshot) -> str:
    """Combined, lowercased robots meta + X-Robots-Tag directives."""
    return f"{snapshot.robots_meta or ''},{snapshot.x_robots_tag or ''}".lower()


def build_ai_access_report(
    snapshot: PageSnapshot,
    site_files: SiteFiles,
    query: str = "",
) -> AIAccessReport:
    """
    Evaluate AI-access signals for a page. Pure: no network access.

    Args:
        snapshot: PageSnapshot from snapshot_url()/parse_snapshot()
        site_files: SiteFiles from fetch_site_files()
        query: Optional target topic (sharpens definition-near-top detection)

    Returns:
        AIAccessReport with issues sorted high -> low severity
    """
    bot_access = evaluate_robots(site_files.robots_txt, snapshot.final_url)
    blocked = tuple(bot for bot, allowed in bot_access.items() if allowed is False)

    directives = _directives(snapshot)
    noindex = "noindex" in directives or "none" in directives.split(",")
    nosnippet = "nosnippet" in directives or "max-snippet:0" in directives.replace(" ", "")

    types = set(snapshot.schema_types)
    has_article = bool(types & {"Article", "NewsArticle", "BlogPosting", "TechArticle"})
    has_org = bool(types & {"Organization", "Corporation", "LocalBusiness", "NewsMediaOrganization"})
    has_person = "Person" in types
    has_faq = "FAQPage" in types

    csr_suspect = (snapshot.word_count < CSR_WORD_THRESHOLD
                   and snapshot.script_count >= CSR_SCRIPT_THRESHOLD)

    issues = []
    search_blocked = [b for b in blocked if b in SEARCH_CRITICAL_BOTS]
    if noindex:
        issues.append(AccessIssue("high", "Page is noindex: search and AI answer engines will not surface it."))
    if search_blocked:
        issues.append(AccessIssue(
            "high", f"robots.txt blocks AI search crawlers: {', '.join(search_blocked)}."))
    if nosnippet:
        issues.append(AccessIssue(
            "high", "nosnippet / max-snippet:0 prevents quoting the page in AI Overviews and snippets."))
    if csr_suspect:
        issues.append(AccessIssue(
            "high", f"Only {snapshot.word_count} words in server HTML with "
                    f"{snapshot.script_count} scripts: content is likely client-rendered and "
                    "invisible to crawlers that do not run JavaScript."))
    training_blocked = [b for b in blocked if b not in SEARCH_CRITICAL_BOTS]
    if training_blocked:
        issues.append(AccessIssue(
            "low", f"robots.txt blocks training crawlers: {', '.join(training_blocked)} "
                   "(limits long-term model knowledge, not live AI search)."))
    if not snapshot.schema_types:
        issues.append(AccessIssue("medium", "No JSON-LD schema found."))
    elif not (has_org or has_person):
        issues.append(AccessIssue(
            "medium", "Schema lacks Organization/Person entities that tie content to a known publisher or author."))
    if site_files.robots_txt is None:
        issues.append(AccessIssue("low", "robots.txt missing or unreachable; AI crawler access is unspecified."))
    if not site_files.llms_txt_present:
        issues.append(AccessIssue("low", "No llms.txt (low impact today; adoption by AI engines is limited)."))

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: severity_rank[i.severity])

    return AIAccessReport(
        bot_access=bot_access,
        blocked_bots=blocked,
        llms_txt_present=site_files.llms_txt_present,
        noindex=noindex,
        nosnippet=nosnippet,
        schema_types=snapshot.schema_types,
        has_article_schema=has_article,
        has_organization_schema=has_org,
        has_person_schema=has_person,
        has_faq_schema=has_faq,
        client_rendered_suspect=csr_suspect,
        content_signals=extract_content_signals(snapshot.main_markdown, query),
        issues=tuple(issues),
    )


def check_ai_access(snapshot: PageSnapshot, query: str = "") -> AIAccessReport:
    """
    Fetch site files for the snapshot's origin and build the access report.

    Raises:
        SnapshotError: If the snapshot URL fails validation
    """
    return build_ai_access_report(snapshot, fetch_site_files(snapshot.final_url), query)
