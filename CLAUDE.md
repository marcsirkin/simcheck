# SimCheck - Code Standards

## Project Overview
**Name:** SimCheck
**Description:** Page/site quality rating (Google QRG) + LLM visibility + query-to-document semantic coverage, for GEO/AI SEO
**Tech Stack:** Python 3.10+, sentence-transformers (BAAI/bge-base-en-v1.5), numpy, streamlit, beautifulsoup4, langchain-typesafe (Jev), openai SDK → OpenRouter
**Version:** 2.0.0

## Codebase Structure
```
simcheck/
├── config.py               # API keys from ~/.config/simcheck/.env (chmod 600, outside repo); never print values
├── core/                   # v1, offline: Content Match
│   ├── models.py, chunker.py, embeddings.py, similarity.py, engine.py
│   ├── diagnostics.py      # CCS, section analysis
│   ├── recommendations.py, geo.py (intent, content signals, action plan)
│   └── readiness.py        # SimScore
├── quality/                # v2, network: page/site quality + LLM visibility
│   ├── snapshot.py         # validated fetch (SSRF-safe redirects, 5MB cap, WAF detection) + extraction
│   ├── ai_access.py        # robots.txt per AI bot (search vs training), noindex, CSR, schema, llms.txt
│   ├── rubric.py           # QRG rubric as data (RUBRIC_VERSION); 9-point slider
│   ├── classifier.py       # Jev / Claude / Hybrid / Fake backends; make_classifier (default "jev")
│   ├── llm_client.py       # OpenRouter client (Claude JSON-schema, citation probes); MODELS constants
│   ├── page_quality.py     # rate_page: QRG gates + band + PQ score
│   ├── probe.py            # Perplexity citation probes
│   ├── report.py           # deterministic headline/summary/fixes for the Report tab
│   ├── site.py             # sitemap discovery, stratified sampling, audit
│   ├── export.py           # shareable HTML report, JSON, CSV
│   └── evaluation.py       # golden-set metrics + offline hybrid simulation
├── tests/                  # 542 tests, no network (fakes + fixtures/)
ui/quality_views.py         # Streamlit views for Report / Page Quality / LLM Visibility / Site Audit
app.py                      # Streamlit router: header, URL bar, 5 tabs; Content Match = v1 flow
eval/run_eval.py            # golden-set agreement (eval/data/ gitignored: client sites)
docs/reference/             # Google QRG PDF/text (gitignored, re-fetch steps in README)
```

## Features

### Feature 1: Core Semantic Comparison Engine
- `compare_query_to_document(query, document, chunking_strategy, chunking_config)` -> `ComparisonResult`
- Supports flat and hierarchical chunking (MARKDOWN, HTML, AUTO strategies)
- Returns max/avg similarity, per-chunk scores, model metadata

### Feature 2: Chunk-Level Diagnostics
- `create_diagnostic_report(result)` -> `DiagnosticReport`
- Sorting (by similarity, by position), filtering (by threshold, custom predicates)
- Heatmap-ready normalized scores, summary statistics
- Section-level analysis for hierarchical documents

### Feature 3: Streamlit Playground UI
- Single-page flow: hero input card, SimScore+CCS banner, drift map, action plan, diagnostics expander
- URL fetcher (convert webpage to Markdown locally via markitdown)
- DKIM example pre-loaded on first visit; "Load example" / "Clear content" buttons
- Chunking strategy selector; GEO intent selector with AI-answer-type labels + auto-detect caption
- Drift map: clickable per-chunk bars in document order (colorblind-safe palette);
  click opens Detailed Diagnostics and smooth-scrolls to that chunk
- Similarity metrics, section analysis, debug panel
- Embedding model warmed at startup (st.cache_resource)

### Streamlit implementation notes (hard-won)
- Writing to a widget's session-state key after the widget renders raises
  StreamlitAPIException — use a pending flag consumed at the top of the render
  function (see load_example_pending / clear_content_pending in app.py)
- st.markdown sanitizes <script>; interactive HTML must go through
  components.html (drift map does this)
- scrollIntoView called from a component iframe on a parent-page element
  silently no-ops in Chrome — scroll section[data-testid="stMain"] directly
- Newlines inside an HTML string passed to st.markdown terminate the HTML
  block mid-element — collapse whitespace first (ui/quality_views._html)
- Escape every page-derived string before st.markdown (titles/URLs/hosts come
  from third-party sites); use esc() in ui/quality_views.py
- Widget keys may be written earlier in the same run, before the widget
  renders: the URL bar's Analyze fills Content Match's query/document this way
- requests: never touch response.apparent_encoding after a streamed read
  ("content already consumed"); use snapshot.decode_body

### Feature 4: Concept Coverage Score (CCS)
- Weighted 0-100 score: Strong=1.0, Moderate=0.6, Weak=0.2, Off-topic=0.0
- Formula: `CCS = (sum of weighted chunks / total chunks) × 100`
- Interpretation bands: 80+ Strong, 60-79 Moderate, 40-59 Weak, <40 Low
- Query-length-aware thresholds: 1-2 word queries use SHORT_QUERY_THRESHOLDS
  (0.72/0.60/0.42) because bare-keyword cosine scores run systematically lower
  than phrase queries; standard set is 0.80/0.65/0.45

### Feature 5: Hierarchical Chunking
- Three-tier hierarchy: MACRO (H2), MICRO (H3), ATOMIC (paragraphs)
- Strategies: FLAT (sentence-based), MARKDOWN, HTML, AUTO (auto-detect)
- Configurable via `ChunkingConfig` (min/max words per level)

### Feature 6: CCS Improvement Recommendations
- `generate_recommendations(report)` -> `RecommendationReport`
- Types: REWRITE_OFF_TOPIC, STRENGTHEN_WEAK, EXPAND_STRONG, RESTRUCTURE_SECTION, REMOVE_DILUTION
- Prioritized (HIGH/MEDIUM/LOW), chunk-specific, with estimated CCS improvement

### Feature 7: GEO Action Plan
- `generate_geo_next_steps(report, document, intent_override)` -> `GeoNextStepsReport`
- Intent detection: informational, how_to, commercial (auto or override)
- Content signal extraction (headings, links, FAQ, TL;DR, steps, examples, etc.)
- Prioritized editor-friendly checklist with time estimates

### Feature 8: SimScore (LLM Readiness)
- `compute_readiness_score(report, signals, intent)` -> `ReadinessScore`
- Composite 0-100: coverage (CCS, 50%) + structure (20%) + evidence (15%) + answerability (15%)
- Bands: 80+ AI-ready, 60-79 Nearly ready, 40-59 Needs work, <40 Not ready

### Feature 9: Page snapshot + AI access (`snapshot.py`, `ai_access.py`)
- Deterministic, no keys. Bot-protected sites raise FetchBlockedError (paste-HTML fallback)
- Main content: semantic container must hold ≥40% of body words; chrome-in-header fallback

### Feature 10: Page Quality rating (`rubric.py`, `classifier.py`, `page_quality.py`)
- 15 QRG questions, one parallel Jev call; QRG gates override the model
- <50 server-rendered words → Unrated (crawlability, not quality)

### Feature 11: Golden-set evaluation (`evaluation.py`, `eval/run_eval.py`)
- Don't show tool ratings for golden URLs before they are hand-rated (anchoring)

### Feature 12: v2 UI (`ui/quality_views.py`, `app.py`)
- One URL, tabs: Report | Page Quality | LLM Visibility | Content Match | Site Audit
- Look: Geist, white, thin rules, no gradients/shadows/emoji/AI tropes

### Feature 13: Citation probes (`probe.py`) — Perplexity Sonar via OpenRouter, click-only, cost shown first

### Feature 14: Site audit (`site.py`) — sitemaps read newest-first by <lastmod> → recent-pages pool (12 mo default) → stratified sample → Jev ratings → sortable grid + CSV

## Code Standards

### Naming Conventions
- Functions: `snake_case` (e.g., `chunk_document`, `embed_text`)
- Classes: `PascalCase` (e.g., `Chunk`, `ComparisonResult`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CHUNK_TOKENS`, `COVERAGE_WEIGHTS`)
- Private functions: `_leading_underscore`
- Enums: `PascalCase` class, `UPPER_SNAKE_CASE` values (e.g., `ChunkLevel.MACRO`)

### Documentation
- All public functions must have docstrings
- Include type hints for all function parameters and returns
- Add inline comments for non-obvious logic

### Error Handling
- Use explicit, descriptive exceptions (ChunkingError, EmbeddingError, ComparisonError)
- Never silently swallow errors
- Validate inputs at module boundaries

### Testing
- Unit tests for all core logic (542 tests); no test may call the network
- Test edge cases explicitly
- Use pytest conventions

## Development Commands
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
pytest simcheck/tests/ -v
streamlit run app.py
python eval/run_eval.py          # golden-set agreement (uses cached answers)
git config core.hooksPath .githooks   # gitleaks pre-commit (brew install gitleaks)
```

## Key Architecture Decisions
- **sentence-transformers** (not Ollama): local-only, no server dependency, better Python integration
- **BAAI/bge-base-en-v1.5**: 768-dim, strong MTEB benchmarks, good accuracy/speed balance
- **~150 token chunks** (flat mode): granular enough for drift detection, within model limits
- **~4 chars/token heuristic**: avoids tokenizer dependency, accurate enough for chunking
- **In-memory only**: no persistence, no vector DB, session resets on reload
- **Thin UI layer**: all logic in `simcheck.core` / `simcheck.quality`, Streamlit just renders
- **Jev for typed rubric answers, Claude only on demand**: Jev ~0.3s/page; Claude changes bands on most pages but isn't yet shown to be more right (eval pending)
- **Jev direct, everything else via OpenRouter**: OpenRouter only offers jev-router (a router, not the classifier)
- **Deterministic report copy**: headline/summary/fixes are templates over measured signals, not LLM text
- **Sitemaps parsed by regex over <loc>**: untrusted XML, avoids entity expansion

## Current Status
**Features Complete:** 1-14
**Test Count:** 542 passing
**Status:** v2.0.0 on branch `feature/quality-visibility` — QRG page rating (Jev), AI access, citation probes, site audit, tabbed UI with shareable report
**Next:** rate golden set → run eval → decide Jev vs hybrid default; citation-gap analysis (compare cited competitor pages) + earned/owned source-type question
