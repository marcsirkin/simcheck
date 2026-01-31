# SimCheck - Code Standards

## Project Overview
**Name:** SimCheck
**Description:** Local query-to-document cosine similarity analyzer
**Tech Stack:** Python 3.10+, sentence-transformers, numpy, streamlit

## Codebase Structure
```
simcheck/
├── core/               # Backend modules (Features 1 + 2)
│   ├── models.py       # Data structures (Chunk, ComparisonResult)
│   ├── chunker.py      # Document chunking logic
│   ├── embeddings.py   # Embedding generation (sentence-transformers)
│   ├── similarity.py   # Cosine similarity calculations
│   ├── engine.py       # Main comparison orchestration (Feature 1)
│   └── diagnostics.py  # Chunk-level diagnostics (Feature 2)
├── tests/              # Unit tests (165 tests)
│   ├── test_models.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_similarity.py
│   ├── test_engine.py
│   └── test_diagnostics.py
└── __init__.py

app.py                  # Streamlit UI (Feature 3)
requirements.txt
```

## Features

### Feature 1: Core Semantic Comparison Engine
- `compare_query_to_document(query, document)` -> `ComparisonResult`
- Chunking, embedding, cosine similarity computation
- Returns max/avg similarity, per-chunk scores

### Feature 2: Chunk-Level Diagnostics
- `create_diagnostic_report(result)` -> `DiagnosticReport`
- Sorting (by similarity, by position)
- Filtering (by threshold, custom predicates)
- Heatmap-ready normalized scores
- Summary statistics

### Feature 3: Streamlit Playground UI
- Interactive text inputs for query and document
- Explicit "Analyze Document" action
- Summary results display
- Per-chunk diagnostics table with sorting
- Debug panel for inspection

## Code Standards

### Naming Conventions
- Functions: `snake_case` (e.g., `chunk_document`, `embed_text`)
- Classes: `PascalCase` (e.g., `Chunk`, `ComparisonResult`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CHUNK_SIZE`)
- Private functions: `_leading_underscore`

### Documentation
- All public functions must have docstrings
- Include type hints for all function parameters and returns
- Add inline comments for non-obvious logic

### Error Handling
- Use explicit, descriptive exceptions
- Never silently swallow errors
- Validate inputs at module boundaries

### Testing
- Unit tests for all core logic
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

## Current Status
**Features Complete:** 1, 2, 3
**Test Count:** 165 passing
**Status:** Core functionality complete
