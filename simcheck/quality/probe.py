"""
Live LLM citation probes (Feature 13).

Asks web-grounded AI engines real questions and checks whether the target
page or its domain is cited. Costs money per query, so the UI runs it only
on an explicit click and shows the estimate first.

Engines go through OpenRouter. Perplexity Sonar is the default because it
returns full, ordered citation lists natively (the spike: 14-19 per answer
at ~$0.005). OpenRouter's search approximates but does not equal the
consumer ChatGPT / Perplexity / AI Mode surfaces: treat results as a
directional signal.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from simcheck.quality.llm_client import MODELS, LLMError, OpenRouterClient


MAX_PROBE_QUERIES = 10
SHORT_TOPIC_WORDS = 4
PROBE_ENGINES = {
    "perplexity": {"model": MODELS["probe_perplexity"], "label": "Perplexity Sonar", "est_cost": 0.006},
}
DEFAULT_ENGINE = "perplexity"


class ProbeError(Exception):
    """Raised for invalid probe input or when every probe call fails."""


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one query on one engine."""
    query: str
    engine: str
    cited_urls: tuple
    cited_hosts: tuple       # normalized (no www.), in citation order
    url_cited: bool          # the exact target page
    domain_cited: bool       # any page on the target domain
    position: Optional[int]  # 1-based position of first target-domain citation
    brand_mentioned: bool    # brand name appears in the answer text
    error: Optional[str] = None


@dataclass(frozen=True)
class ProbeReport:
    """All probe results for a target page."""
    target_url: str
    target_domain: str
    brand: str
    results: tuple
    cost: Optional[float]

    @property
    def completed(self) -> tuple:
        """Results that returned an answer."""
        return tuple(r for r in self.results if r.error is None)

    @property
    def cited_count(self) -> int:
        """Answers citing the target domain."""
        return sum(1 for r in self.completed if r.domain_cited)

    def top_competitors(self, n: int = 5) -> list:
        """
        Most-cited other domains across all answers.

        Returns:
            [(host, number of answers citing it)], most frequent first
        """
        counts = Counter()
        for r in self.completed:
            counts.update({h for h in r.cited_hosts if h != self.target_domain})
        return counts.most_common(n)


def normalize_host(url_or_host: str) -> str:
    """Lowercased host without a leading www."""
    host = urlparse(url_or_host).netloc if "//" in url_or_host else url_or_host
    host = host.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _normalize_url(url: str) -> str:
    """Comparable URL: host without www, path without trailing slash, no query/fragment."""
    parsed = urlparse(url)
    return normalize_host(parsed.netloc) + (parsed.path.rstrip("/") or "/")


def brand_from_domain(domain: str) -> str:
    """Best-guess brand name from a domain (healthline.com -> healthline)."""
    parts = normalize_host(domain).split(".")
    # Skip common second-level suffixes like co.uk / com.au
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov"):
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[0]


def default_queries(query: Optional[str], title: str = "") -> list:
    """
    Seed probe questions from the target query (user edits them in the UI).

    Args:
        query: Target query from the analysis
        title: Page title, used when no query is given

    Returns:
        Up to 3 distinct questions
    """
    base = (query or "").strip() or re.split(r"[:|\-–]", title or "")[0].strip()
    if not base:
        return []
    candidates = [base]
    # Only wrap short topic phrases; "What is high blood pressure symptoms
    # and treatment?" reads badly, so longer queries are used as typed.
    if not base.endswith("?") and len(base.split()) <= SHORT_TOPIC_WORDS:
        candidates += [f"What is {base}?", f"What should I know about {base}?"]
    seen, out = set(), []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:3]


def estimate_cost(n_queries: int, engine: str = DEFAULT_ENGINE) -> float:
    """Rough USD cost for a probe run."""
    return n_queries * PROBE_ENGINES[engine]["est_cost"]


def evaluate_answer(query: str, engine: str, text: str, urls: list,
                    target_url: str, brand: str) -> ProbeResult:
    """
    Score one answer against the target. Pure: no network.

    Args:
        query: Question asked
        engine: Engine key
        text: Answer text
        urls: Cited URLs in order
        target_url: Page being checked
        brand: Brand name to look for in the text

    Returns:
        ProbeResult
    """
    target_domain = normalize_host(target_url)
    target_norm = _normalize_url(target_url)
    hosts = tuple(normalize_host(u) for u in urls)
    position = next((i + 1 for i, h in enumerate(hosts)
                     if h == target_domain or h.endswith("." + target_domain)), None)
    brand_re = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE) if brand else None
    return ProbeResult(
        query=query,
        engine=engine,
        cited_urls=tuple(urls),
        cited_hosts=hosts,
        url_cited=any(_normalize_url(u) == target_norm for u in urls),
        domain_cited=position is not None,
        position=position,
        brand_mentioned=bool(brand_re and brand_re.search(text or "")),
    )


def run_probes(
    queries: list,
    target_url: str,
    llm: OpenRouterClient,
    engine: str = DEFAULT_ENGINE,
    brand: Optional[str] = None,
) -> ProbeReport:
    """
    Ask each query once on the engine and check for target citations.

    A failed query is recorded with its error; the run fails only if every
    query fails.

    Args:
        queries: Questions (1..MAX_PROBE_QUERIES, blanks ignored)
        target_url: Page to look for
        llm: OpenRouterClient
        engine: Key in PROBE_ENGINES
        brand: Brand name (defaults to one derived from the domain)

    Returns:
        ProbeReport

    Raises:
        ProbeError: On invalid input or if all queries fail
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ProbeError("Add at least one question to probe.")
    if len(queries) > MAX_PROBE_QUERIES:
        raise ProbeError(f"At most {MAX_PROBE_QUERIES} questions per run.")
    if engine not in PROBE_ENGINES:
        raise ProbeError(f"Unknown engine {engine!r}.")

    domain = normalize_host(target_url)
    brand = brand or brand_from_domain(domain)
    model = PROBE_ENGINES[engine]["model"]

    results, total_cost, any_cost = [], 0.0, False
    for q in queries:
        try:
            text, urls, cost = llm.chat_with_citations(model, q)
        except LLMError as e:
            results.append(ProbeResult(q, engine, (), (), False, False, None, False, error=str(e)))
            continue
        if cost is not None:
            total_cost += cost
            any_cost = True
        results.append(evaluate_answer(q, engine, text, urls, target_url, brand))

    if all(r.error for r in results):
        raise ProbeError(f"All probes failed: {results[0].error}")
    return ProbeReport(target_url=target_url, target_domain=domain, brand=brand,
                       results=tuple(results), cost=total_cost if any_cost else None)
