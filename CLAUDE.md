# SimCheck - Code Standards

## Project Overview
**Name:** SimCheck
**Description:** Local query-to-document cosine similarity analyzer for AI visibility (GEO/AI SEO) optimization
**Tech Stack:** Python 3.10+, sentence-transformers (BAAI/bge-base-en-v1.5), numpy, streamlit
**Version:** 1.1.0

## Codebase Structure
```
simcheck/
├── core/                   # Backend modules
│   ├── models.py           # Data structures (Chunk, ChunkLevel, ChunkSimilarity, ComparisonResult, SIMILARITY_THRESHOLDS)
│   ├── chunker.py          # Document chunking (flat + hierarchical: MARKDOWN, HTML, AUTO)
│   ├── embeddings.py       # Embedding generation (sentence-transformers, BAAI/bge-base-en-v1.5)
│   ├── similarity.py       # Cosine similarity calculations
│   ├── engine.py           # Main comparison orchestration (compare_query_to_document)
│   ├── diagnostics.py      # Chunk-level diagnostics, CCS scoring, section analysis
│   ├── recommendations.py  # CCS improvement recommendations (prioritized, chunk-specific)
│   ├── geo.py              # GEO/AI-SEO action plan (intent detection, content signals, next steps)
│   └── readiness.py        # SimScore: composite LLM readiness metric (Feature 8)
├── tests/                  # Unit tests (272 tests)
│   ├── test_models.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_similarity.py
│   ├── test_engine.py
│   ├── test_diagnostics.py
│   ├── test_hierarchical_chunker.py
│   ├── test_recommendations.py
│   └── test_geo.py
└── __init__.py             # Public API exports

app.py                      # Streamlit UI (two tabs: Action Plan + Diagnostics)
requirements.txt
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
  block mid-element — collapse whitespace first

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
- Reportable headline metric; component breakdown shown in UI banner

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
- Unit tests for all core logic (292 tests)
- Test edge cases explicitly
- Use pytest conventions

## Development Commands
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest simcheck/tests/ -v

# Run Streamlit app
streamlit run app.py
```

## Key Architecture Decisions
- **sentence-transformers** (not Ollama): local-only, no server dependency, better Python integration
- **BAAI/bge-base-en-v1.5**: 768-dim, strong MTEB benchmarks, good accuracy/speed balance
- **~150 token chunks** (flat mode): granular enough for drift detection, within model limits
- **~4 chars/token heuristic**: avoids tokenizer dependency, accurate enough for chunking
- **In-memory only**: no persistence, no vector DB, session resets on reload
- **Thin UI layer**: all semantic logic in `simcheck.core`, Streamlit just renders

## Current Status
**Features Complete:** 1, 2, 3, 4, 5, 6, 7, 8
**Test Count:** 292 passing
**Status:** v1.3.0 — SimScore readiness metric, clickable drift map, DKIM example pre-load, clear-content button, intent labels clarified
