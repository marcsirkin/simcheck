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
│   └── geo.py              # GEO/AI-SEO action plan (intent detection, content signals, next steps)
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
- Two-tab layout: Action Plan (GEO next steps) + Diagnostics (chunk-level detail)
- URL fetcher (convert webpage to Markdown via urltomarkdown.com)
- Chunking strategy selector, GEO intent override
- CCS score display, similarity metrics, section analysis, debug panel

### Feature 4: Concept Coverage Score (CCS)
- Weighted 0-100 score: Strong=1.0, Moderate=0.6, Weak=0.2, Off-topic=0.0
- Formula: `CCS = (sum of weighted chunks / total chunks) × 100`
- Interpretation bands: 80+ Strong, 60-79 Moderate, 40-59 Weak, <40 Low

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
- Unit tests for all core logic (272 tests)
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
**Features Complete:** 1, 2, 3, 4, 5, 6, 7
**Test Count:** 271 passing
**Status:** v1.1.0 — Modern UI overhaul (Atlassian-inspired card layout, colored CCS banners, single-column action plan)
