# Product Requirements Document: SimCheck

## Project Overview

**Name:** SimCheck - Local Query-to-Document Cosine Similarity Analyzer

**Problem Statement:**
As a consultant and practitioner, I need a local-first tool that allows me to:
- Take a concept or term (e.g., "Major League Baseball")
- Take a block of text (e.g., a blog post or page copy)
- Analyze how strongly and *where* that text semantically aligns with the concept
- Build intuition around cosine similarity, embeddings, and content drift
- Use this insight to advise myself and clients

This tool is for analysis and learning, not client-facing demos.

**Target User:**
- Single expert user (you)
- Comfortable running local scripts
- Comfortable reading numeric output and distributions
- Wants transparency over abstraction

**Success Criteria:**
- Chunk a document and see per-chunk similarity scores against a query term
- Identify where content drifts off-topic within a document
- Build intuition for what similarity scores mean in practice
- Tool runs fully offline with no external dependencies

---

## Non-Goals (Explicit)

To keep this tight, the application will **not**:
- Be deployed publicly
- Support authentication or multi-user
- Crawl websites automatically (paste-only for v1)
- Optimize for performance at scale
- Generate recommendations or rewritten content
- Use a vector database (not needed for v1)

---

## Core Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Step 1    │     │   Step 2    │     │   Step 3    │
│   Chunk     │ ──▶ │   Embed     │ ──▶ │   Query     │
│  Document   │     │   Chunks    │     │  Similarity │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Step 1: Chunk Document**
- Triggered by: User clicks "Chunk Document" or runs CLI command
- Input: Pasted document text
- Output: List of chunks with preview, chunk count

**Step 2: Embed Chunks**
- Triggered: Automatically after chunking (or explicit in CLI)
- Input: Chunked text
- Output: Cached chunk embeddings, confirmation message

**Step 3: Run Query Similarity**
- Triggered by: User clicks "Run Query" or runs CLI command
- Input: Query term/phrase
- Output: Per-chunk similarity scores, max, average

---

## Core Features

### Feature 1: Document Chunking
**Description:** Split a pasted document into token-based chunks for analysis.

**User Story:** As an analyst, I want to chunk a document so that I can analyze semantic alignment at a granular level.

**Acceptance Criteria:**
- [ ] Text input area for pasting document
- [ ] "Chunk Document" button triggers chunking
- [ ] Default chunk size: 300-400 tokens
- [ ] Preserve chunk order (position matters)
- [ ] Display chunk count after processing
- [ ] Preview first N chunks (collapsible)
- [ ] Store raw chunk text for later display

**Technical Notes:**
- Use tiktoken for token counting and splitting
- Split on token boundaries, not mid-word
- Handle edge cases: very short documents, single chunk

---

### Feature 2: Chunk Embedding
**Description:** Convert each chunk into a vector embedding using Ollama.

**User Story:** As an analyst, I want chunks embedded so that I can compute semantic similarity against my query.

**Acceptance Criteria:**
- [ ] Automatically triggered after chunking completes
- [ ] Use consistent embedding model across all chunks
- [ ] Display embedding dimensionality
- [ ] Show progress indicator during embedding
- [ ] Confirm when document is "indexed" (ready for queries)
- [ ] Cache embeddings in memory (no re-embed on new query)

**Technical Notes:**
- Use Ollama `embed` API
- Default model: `nomic-embed-text`
- Store embeddings as numpy arrays
- No disk persistence needed for v1

---

### Feature 3: Query Similarity Analysis
**Description:** Embed a query term and compute cosine similarity against all document chunks.

**User Story:** As an analyst, I want to see how each chunk aligns with my query so that I can identify content drift.

**Acceptance Criteria:**
- [ ] Text input for query term/phrase
- [ ] "Run Query" button triggers similarity calculation
- [ ] Embed query using same model as chunks
- [ ] Compute cosine similarity: query vs each chunk
- [ ] Display results in document order (chunk position matters)
- [ ] Show max similarity (strongest alignment)
- [ ] Show average similarity (overall topical coverage)
- [ ] Per-chunk similarity table with scores

**Output Format:**
```
Max Similarity:  0.88 (Chunk #2)
Avg Similarity:  0.61

Chunk #    Similarity    Interpretation
1          0.82          Strong
2          0.88          Strong
3          0.41          Off-topic
4          0.33          Off-topic
```

**Technical Notes:**
- Cosine similarity via numpy: `dot(a, b) / (norm(a) * norm(b))`
- Format scores to 2 decimal places
- Optional: allow sorting by score or position

---

### Feature 4: Interpretation Thresholds
**Description:** Provide heuristic interpretation of similarity scores.

**User Story:** As a learner, I want guidance on what scores mean so that I can build intuition.

**Acceptance Criteria:**
- [ ] Display interpretation label next to each score
- [ ] Thresholds are configurable (not hardcoded magic)
- [ ] Clear disclaimer: "These are heuristics, not truth"

**Default Thresholds:**
| Score Range | Interpretation |
|-------------|----------------|
| ≥ 0.80 | Strong semantic alignment |
| 0.65–0.80 | Moderate alignment |
| 0.45–0.65 | Weak / partial alignment |
| < 0.45 | Likely off-topic |

**Technical Notes:**
- Store thresholds in config dict
- Color coding in UI: green/yellow/orange/red

---

### Feature 5: Model Selection
**Description:** Choose from available Ollama embedding models.

**User Story:** As an analyst, I want to switch models so that I can compare results across different embeddings.

**Acceptance Criteria:**
- [ ] Dropdown/selector for model choice
- [ ] Default model works out of the box
- [ ] Switching models requires re-embedding chunks
- [ ] Clear warning when model changes

**Available Models (v1):**
| Model | Notes |
|-------|-------|
| `nomic-embed-text` (default) | Good general purpose, fast |
| `mxbai-embed-large` | Higher quality, slower |

**Technical Notes:**
- Query available models via `ollama list`
- Validate model exists before embedding

---

## Technical Architecture

### Tech Stack
**Language:** Python 3.10+
**Embeddings:** Ollama (local)
**Chunking:** tiktoken
**Math:** NumPy
**UI (Day 2+):** Streamlit

### Dependencies
```
ollama
tiktoken
numpy
streamlit  # Day 2+
```

### Application Structure
```
simcheck/
├── cli.py                 # CLI entry point (Day 1)
├── app.py                 # Streamlit UI (Day 2)
├── core/
│   ├── __init__.py
│   ├── chunker.py         # Token-based text chunking
│   ├── embeddings.py      # Ollama embedding calls
│   └── similarity.py      # Cosine similarity calculations
├── config.py              # Thresholds, defaults
├── tests/
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   └── test_similarity.py
├── requirements.txt
└── README.md
```

### Architecture Flow
```
Input Text
   ↓
Chunker (tiktoken)
   ↓
Embedding Engine (Ollama)
   ↓
Cached Chunk Vectors (numpy arrays in memory)
   ↓
Query Embedding (Ollama)
   ↓
Cosine Similarity (numpy)
   ↓
Metrics + Output
```

### External Dependencies
- **Ollama:** Must be installed and running locally
- **Models:** User must have pulled embedding models (`ollama pull nomic-embed-text`)

### Storage
**Choice:** In-memory only
**Notes:** No vector DB needed. Embeddings cached as numpy arrays during session. No persistence between runs.

---

## UI/UX Design

### Development Phases

**Day 1: CLI/Notebook**
- Validate core logic without UI overhead
- Print output to terminal
- Fast iteration on chunking/embedding/similarity

**Day 2: Streamlit UI**
- Add visual interface
- Buttons for each workflow step
- Results table with color coding

### CLI Interface (Day 1)
```bash
# Chunk and embed a document
python cli.py chunk --file document.txt
python cli.py chunk --text "paste text here"

# Run query against embedded document
python cli.py query "Major League Baseball"

# Full pipeline
python cli.py analyze --text "..." --query "Major League Baseball"
```

### Streamlit Layout (Day 2)
```
┌─────────────────────────────────────────────────────────┐
│  SimCheck                                   [Model ▼]   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Document                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  [Paste document text here...]                   │   │
│  │                                                   │   │
│  └─────────────────────────────────────────────────┘   │
│  [ Chunk & Embed Document ]                             │
│                                                         │
│  Chunks: 4 chunks (1,247 tokens total)                 │
│  Status: ✓ Document indexed                             │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query Term                                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Major League Baseball                           │   │
│  └─────────────────────────────────────────────────┘   │
│  [ Run Query ]                                          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Results                                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Max Similarity:  0.88 (Chunk #2)               │   │
│  │  Avg Similarity:  0.61                           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Per-Chunk Analysis                                     │
│  ┌──────┬────────────┬────────────────┬────────────┐   │
│  │ #    │ Similarity │ Interpretation │ Preview    │   │
│  ├──────┼────────────┼────────────────┼────────────┤   │
│  │ 1    │ 0.82       │ Strong         │ "The MLB..."│   │
│  │ 2    │ 0.88       │ Strong         │ "Baseball..."│  │
│  │ 3    │ 0.41       │ Off-topic      │ "However..."│   │
│  │ 4    │ 0.33       │ Off-topic      │ "In other..."│  │
│  └──────┴────────────┴────────────────┴────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Interactions
- **Explicit steps:** User triggers each phase (chunk, query)
- **Button-triggered:** No auto-processing on paste
- **Progress indicators:** Show embedding progress
- **Inline feedback:** Status messages below each section

### Error Handling
- Inline error messages: "Ollama not running", "Model not found"
- Graceful degradation: suggest fallback model if preferred unavailable
- Clear guidance: "Run `ollama pull nomic-embed-text` to install"

---

## Development Checklist

### Day 1: Core Logic (CLI)
- [ ] Set up project structure
- [ ] Implement chunker.py with tiktoken
- [ ] Implement embeddings.py with Ollama
- [ ] Implement similarity.py with numpy
- [ ] Create cli.py with basic commands
- [ ] Test: chunk a document, print chunks
- [ ] Test: embed chunks, print dimensions
- [ ] Test: query similarity, print max/avg
- [ ] Write unit tests for core modules

### Day 2: Streamlit UI
- [ ] Create app.py with basic layout
- [ ] Add document text area
- [ ] Add "Chunk & Embed" button
- [ ] Add query input
- [ ] Add "Run Query" button
- [ ] Display results table
- [ ] Add model selector
- [ ] Add interpretation colors

### Day 3: Polish
- [ ] Tune chunk size for best results
- [ ] Calibrate interpretation thresholds
- [ ] Test with multiple real documents
- [ ] Add chunk preview expansion
- [ ] Handle edge cases

---

## Trade-offs & Decisions

### Decision Log
| Date | Decision | Rationale | Alternative Considered |
|------|----------|-----------|------------------------|
| 2026-01-30 | Ollama over sentence-transformers | Already installed locally, easy model switching, no Python dep conflicts | sentence-transformers (more models but heavier) |
| 2026-01-30 | CLI first, Streamlit day 2 | Validate logic fast, avoid UI yak-shaving | Streamlit from start (slower iteration) |
| 2026-01-30 | tiktoken for chunking | Precise token control, matches model tokenization | Sentence-based (variable chunk sizes) |
| 2026-01-30 | Explicit chunking step | User sees what's happening, transparency over magic | Auto-chunk on paste (less control) |
| 2026-01-30 | In-memory only | Simplicity, no persistence needed | SQLite/pickle (adds complexity) |
| 2026-01-30 | Query-to-document (1:N) | Matches actual use case: analyze content coverage | Text A vs B comparison (different problem) |

---

## Interpretation Guide

### Understanding Scores
Cosine similarity ranges from -1 to 1, but for normalized embeddings typically 0 to 1:
- **1.0:** Identical meaning (same text)
- **0.8+:** Very strong semantic alignment
- **0.6-0.8:** Related topics, moderate alignment
- **0.4-0.6:** Tangentially related
- **<0.4:** Different topics, likely off-target

### Building Intuition
Run the same query against:
1. A document that's clearly on-topic → expect 0.7-0.9
2. A document that's tangentially related → expect 0.5-0.7
3. A completely unrelated document → expect 0.2-0.4

This calibrates your mental model for what scores mean.

---

## Future Enhancements (v2+)

### v1.1 (Soon)
- [ ] Configurable chunk size via UI
- [ ] Sort results by score or position
- [ ] Flag chunks below threshold
- [ ] Histogram of similarity distribution

### v2 (Later)
- [ ] Multiple queries at once
- [ ] Compare two documents against same query
- [ ] Export results to JSON/CSV
- [ ] Session history (save/load analyses)

### v3+ (Maybe)
- [ ] OpenRouter/cloud embeddings option
- [ ] Batch processing from file
- [ ] API endpoint for scripting

---

## Reference

### Useful Repos/Resources
- **Ollama:** https://github.com/ollama/ollama
- **Embedding search:** `ollama embed cosine similarity` on GitHub
- **Streamlit:** https://github.com/streamlit/streamlit

### Ollama Commands
```bash
# Install/start Ollama
ollama serve

# Pull embedding models
ollama pull nomic-embed-text
ollama pull mxbai-embed-large

# List available models
ollama list
```

### Sample Code Reference
```python
import ollama
import numpy as np

# Embed text
response = ollama.embed(model='nomic-embed-text', input='Hello world')
embedding = np.array(response['embeddings'][0])

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
