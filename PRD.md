# Product Requirements Document: SimCheck

## Project Overview

**Name:** SimCheck - Local Query-to-Document Cosine Similarity Analyzer

**Problem Statement:**
As a consultant and practitioner, I need a local-first tool that allows me to:
- Take a concept or term (e.g., "Major League Baseball")
- Take a block of text (e.g., a blog post or page copy)
- Analyze how strongly and *where* that text semantically aligns with the concept
- Build intuition around cosine similarity, embeddings, and content drift
- Get actionable GEO/AI-SEO recommendations for improving AI visibility
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
- Get a Concept Coverage Score (CCS) quantifying topical alignment
- Receive prioritized, actionable recommendations for improving content
- Get a GEO-oriented action plan tailored to query intent
- Build intuition for what similarity scores mean in practice
- Tool runs fully offline with no external dependencies (except optional URL fetch)

---

## Non-Goals (Explicit)

To keep this tight, the application will **not**:
- Be deployed publicly
- Support authentication or multi-user
- Crawl websites automatically (paste-only or single URL fetch for v1)
- Optimize for performance at scale
- Generate recommendations or rewritten content automatically
- Use a vector database (not needed for v1)

---

## Core Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Step 1    │     │   Step 2    │     │   Step 3    │     │   Step 4    │
│   Chunk     │ ──▶ │   Embed     │ ──▶ │   Query     │ ──▶ │  Diagnose   │
│  Document   │     │   Chunks    │     │  Similarity │     │ + Recommend │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Step 1: Chunk Document**
- Input: Pasted document text (or fetched from URL)
- Strategy: FLAT (sentence-based), MARKDOWN, HTML, or AUTO
- Output: List of chunks with metadata (level, heading, position)

**Step 2: Embed Chunks**
- Automatic after chunking
- Model: BAAI/bge-base-en-v1.5 (768 dims, L2-normalized)
- Output: Cached chunk embeddings as numpy arrays

**Step 3: Run Query Similarity**
- Input: Query term/phrase (entity + intent)
- Output: Per-chunk cosine similarity scores, max, average

**Step 4: Diagnose + Recommend**
- Output: CCS score, chunk diagnostics, improvement recommendations, GEO action plan

---

## Core Features

### Feature 1: Core Semantic Comparison Engine ✅
**Description:** Compare a query/concept against a document and return semantic similarity analysis.

**User Story:** As an analyst, I want to see how each chunk aligns with my query so that I can identify content drift.

**Acceptance Criteria:**
- [x] `compare_query_to_document()` orchestrates full pipeline
- [x] Supports flat and hierarchical chunking strategies
- [x] Returns max/avg similarity, per-chunk scores, model metadata
- [x] Input validation with descriptive error messages
- [x] Batch embedding for efficiency

**Technical Notes:**
- Sentence-boundary chunking (~150 tokens for flat mode)
- sentence-transformers with BAAI/bge-base-en-v1.5
- Cosine similarity via numpy dot product (normalized vectors)

---

### Feature 2: Chunk-Level Diagnostics ✅
**Description:** Transform comparison results into inspectable, sortable, filterable diagnostic structures.

**User Story:** As an analyst, I want detailed per-chunk diagnostics so I can understand exactly why a document scores the way it does.

**Acceptance Criteria:**
- [x] `create_diagnostic_report()` transforms ComparisonResult
- [x] Sorting: by document order, similarity ascending/descending
- [x] Filtering: by threshold, by range, by custom predicate
- [x] Convenience methods: strong_chunks(), off_topic_chunks(), top_n(), bottom_n()
- [x] Heatmap-ready normalized scores (0-1)
- [x] Summary statistics (mean, median, std dev, threshold counts)
- [x] Section-level analysis for hierarchical documents

**Technical Notes:**
- Pure transformation, no recomputation of embeddings
- Normalized scores relative to document min/max

---

### Feature 3: Streamlit Playground UI ✅
**Description:** Interactive web UI for exploring semantic similarity results.

**User Story:** As an analyst, I want a visual interface to explore results, compare content versions, and drill into chunk-level details.

**Acceptance Criteria:**
- [x] Single-page flow: hero input card, CCS banner, action plan, diagnostics expander
- [x] Text input for query (target topic) and document
- [x] URL fetcher (convert webpage to Markdown via markitdown)
- [x] Chunking strategy selector (flat, auto, markdown, html)
- [x] GEO intent override (auto, informational, how_to, commercial)
- [x] "Analyze Document" full-width button triggers full pipeline
- [x] CCS score as colored-accent banner card (green/amber/red/gray by band)
- [x] Similarity metrics (max, avg, chunks, on-topic %)
- [x] Per-chunk diagnostics table with sorting and level filtering
- [x] Best/worst chunks quick access
- [x] Section analysis for hierarchical documents
- [x] Action plan with bordered step cards (single-column, numbered badges)
- [x] Content signal pills in tinted strip
- [x] Debug panel with heatmap data

**Technical Notes:**
- Thin UI layer: all logic in simcheck.core
- Session state management, no persistence between sessions
- Atlassian-inspired design: #F4F5F7 gray canvas, white card zones, #1868DB primary blue
- Uses `st.container(border=True)` for card layout (not raw HTML divs)

---

### Feature 4: Concept Coverage Score (CCS) ✅
**Description:** A weighted 0-100 score measuring how thoroughly a document expresses the target concept.

**User Story:** As an analyst, I want a single score that tells me how well-covered my target concept is across the document.

**Acceptance Criteria:**
- [x] Weighted score formula: CCS = (sum of weighted chunks / total) × 100
- [x] Bucket weights: Strong=1.0, Moderate=0.6, Weak=0.2, Off-topic=0.0
- [x] Interpretation bands: 80+ Strong, 60-79 Moderate, 40-59 Weak, <40 Low
- [x] Single-chunk warning flag

**Default Thresholds:**
| Score Range | Interpretation | CCS Weight |
|-------------|----------------|------------|
| >= 0.80     | Strong         | 1.0        |
| 0.65-0.80   | Moderate       | 0.6        |
| 0.45-0.65   | Weak           | 0.2        |
| < 0.45      | Off-topic      | 0.0        |

---

### Feature 5: Hierarchical Chunking ✅
**Description:** Parse document structure (Markdown/HTML headings) to create a three-tier chunk hierarchy.

**User Story:** As an analyst, I want to see per-section coverage so I can identify which sections need work.

**Acceptance Criteria:**
- [x] Three-tier hierarchy: MACRO (H2), MICRO (H3), ATOMIC (paragraphs)
- [x] Strategies: FLAT, MARKDOWN, HTML, AUTO
- [x] AUTO detects format based on content patterns
- [x] Configurable via ChunkingConfig (min/max words per level)
- [x] Falls back to FLAT when no structure detected
- [x] Section-level aggregated statistics (SectionSummary)

**Chunking Strategies:**
| Strategy | Use Case |
|----------|----------|
| `FLAT`   | Default, sentence-based chunking |
| `MARKDOWN` | Parse `##` and `###` headings |
| `HTML`   | Parse `<h2>` and `<h3>` tags |
| `AUTO`   | Auto-detect format |

---

### Feature 6: CCS Improvement Recommendations ✅
**Description:** Generate prioritized, chunk-specific recommendations for improving CCS.

**User Story:** As an analyst, I want actionable recommendations for improving my document's concept coverage.

**Acceptance Criteria:**
- [x] `generate_recommendations(report)` analyzes diagnostic report
- [x] Recommendation types: REWRITE_OFF_TOPIC, STRENGTHEN_WEAK, EXPAND_STRONG, RESTRUCTURE_SECTION, REMOVE_DILUTION
- [x] Priority levels: HIGH, MEDIUM, LOW
- [x] Each recommendation includes: what, why, how, target chunks, estimated impact
- [x] Example text from strong chunks for weaker sections
- [x] Estimated potential CCS after implementing fixes
- [x] Max 5 recommendations, ranked by priority and impact

---

### Feature 7: GEO Action Plan ✅
**Description:** Generate a GEO-oriented action plan complementing CCS with editor-friendly next steps.

**User Story:** As a content editor, I want a prioritized checklist of what to change, where, and why, so I can improve my page's AI visibility.

**Acceptance Criteria:**
- [x] `generate_geo_next_steps(report, document, intent_override)` produces action plan
- [x] Intent detection: informational, how_to, commercial (auto or override)
- [x] Content signal extraction (headings, links, FAQ, TL;DR, steps, examples, sources, freshness)
- [x] Actions: front-load definition, reduce drift, strengthen weak, add structure, add evidence, add examples, add FAQ, add TL;DR
- [x] Intent-specific actions (how-to: add steps; commercial: add comparison section)
- [x] Each step includes: title, priority, why, how, time estimate, examples, target chunks
- [x] Sorted: HIGH → MEDIUM → LOW, then shortest time first

---

## Technical Architecture

### Tech Stack
**Language:** Python 3.10+
**Embeddings:** sentence-transformers (BAAI/bge-base-en-v1.5, 768 dims)
**Math:** NumPy (cosine similarity via dot product on normalized vectors)
**UI:** Streamlit
**URL Fetch:** markitdown (local conversion, no external API)

### Dependencies
```
sentence-transformers>=2.2.0
numpy>=1.24.0
torch>=2.0.0
pytest>=7.0.0
streamlit>=1.28.0
requests>=2.31.0
```

### Architecture Flow
```
Input Text (paste or URL fetch)
   ↓
Chunker (flat: sentence-boundary | hierarchical: heading-based)
   ↓
Embedding Engine (sentence-transformers, bge-base-en-v1.5)
   ↓
Cached Chunk Vectors (numpy arrays in memory, L2-normalized)
   ↓
Query Embedding (same model)
   ↓
Cosine Similarity (numpy dot product)
   ↓
Diagnostics (CCS, chunk analysis, section analysis)
   ↓
Recommendations + GEO Action Plan
   ↓
Streamlit UI (Action Plan tab + Diagnostics tab)
```

### External Dependencies
- **sentence-transformers:** Downloads model on first use (~400MB for bge-base-en-v1.5)
- **markitdown:** Local URL-to-Markdown conversion (no external API dependency)

### Storage
**Choice:** In-memory only
**Notes:** No vector DB needed. Embeddings cached as numpy arrays during session. No persistence between runs.

---

## UI/UX Design

### Streamlit Layout (v1.1)
```
┌─────────────────────────────────────────────────────────┐
│  SimCheck — semantic coverage analyzer                   │
│  [How to use (collapsible)]                              │
├─────────────────────────────────────────────────────────┤
│  ANALYZE A PAGE                                          │
│  ┌─ Hero Card ────────────────────────────────────────┐ │
│  │  [URL input ________________________] [Fetch]      │ │
│  │  [Target topic ____________________________]       │ │
│  └────────────────────────────────────────────────────┘ │
│  [Document textarea]                                     │
│  [Settings (collapsible)]                                │
│  [======== Analyze Document (full-width) ========]       │
├─────────────────────────────────────────────────────────┤
│  ┌─ CCS Banner (colored accent) ─────────────────────┐ │
│  │  Score | Interpretation | Buckets | Potential      │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌─ Action Plan Card ────────────────────────────────┐ │
│  │  Summary + Content signal pills strip             │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌─ Step 1 ──────────────────────────────────────────┐ │
│  │  [num] Title [PRIORITY] [~X min]                  │ │
│  │  Why · How · Expanders                            │ │
│  └────────────────────────────────────────────────────┘ │
│  ... more steps ...                                      │
│  ▸ Detailed Diagnostics (collapsed expander)             │
└─────────────────────────────────────────────────────────┘
```

---

## Trade-offs & Decisions

### Decision Log
| Date | Decision | Rationale | Alternative Considered |
|------|----------|-----------|------------------------|
| 2026-01-30 | Ollama for embeddings | Already installed locally, easy model switching | sentence-transformers |
| 2026-01-30 | CLI first, Streamlit day 2 | Validate logic fast | Streamlit from start |
| 2026-01-30 | Explicit chunking step | Transparency over magic | Auto-chunk on paste |
| 2026-01-30 | In-memory only | Simplicity | SQLite/pickle |
| 2026-01-30 | Query-to-document (1:N) | Matches actual use case | Text A vs B comparison |
| 2026-02-XX | Switch to sentence-transformers | No server dependency, better Python integration | Ollama (requires running server) |
| 2026-02-XX | BAAI/bge-base-en-v1.5 | Strong MTEB benchmarks, 768-dim, good balance | nomic-embed-text, all-MiniLM-L6-v2 |
| 2026-02-XX | ~4 chars/token heuristic | Avoids tokenizer dependency | tiktoken (heavier, precise) |
| 2026-02-XX | Add CCS scoring | Need single number for content quality | Raw similarity only |
| 2026-02-XX | Add hierarchical chunking | Need section-level analysis for structured docs | Flat only |
| 2026-02-XX | Add GEO action plan | Need editor-friendly, intent-aware next steps | CCS recs only |

---

## Interpretation Guide

### Understanding Scores
Cosine similarity ranges from -1 to 1, but for L2-normalized sentence-transformer embeddings typically 0 to 1:
- **1.0:** Identical meaning (same text)
- **0.8+:** Very strong semantic alignment
- **0.65-0.80:** Related topics, moderate alignment
- **0.45-0.65:** Tangentially related
- **<0.45:** Different topics, likely off-target

### CCS Score
| CCS Range | Interpretation | Meaning |
|-----------|----------------|---------|
| 80-100    | Strong         | Strong topical focus |
| 60-79     | Moderate       | Decent focus; fix weak/off-topic sections |
| 40-59     | Weak           | Weak focus; restructure + improve intro |
| <40       | Low            | Content doesn't answer the topic directly |

### Building Intuition
Run the same query against:
1. A document that's clearly on-topic → expect 0.7-0.9
2. A document that's tangentially related → expect 0.5-0.7
3. A completely unrelated document → expect 0.2-0.4

---

## Future Enhancements (v2+)

### v1.1 (Current) ✅
- [x] Modern UI overhaul (Atlassian-inspired card layout, colored CCS banners)
- [x] Local URL-to-Markdown conversion (markitdown, no external API)
- [x] Single-column action plan with bordered step cards
- [x] Content signal pills in tinted strip
- [x] Hero input card with collapsed labels

### v1.2 (Soon)
- [ ] Configurable chunk size via UI
- [ ] Histogram of similarity distribution
- [ ] Export results to JSON/CSV
- [ ] Compare two drafts side-by-side

### v2 (Later)
- [ ] Multiple queries at once
- [ ] Compare two documents against same query
- [ ] Session history (save/load analyses)
- [ ] Batch URL processing

### v3+ (Maybe)
- [ ] Cloud embedding option (OpenAI, Cohere)
- [ ] API endpoint for scripting
- [ ] Custom threshold configuration in UI
- [ ] Automatic content rewrite suggestions

---

## Session Management

**Test Count:** 271 passing
**Features Complete:** 1, 2, 3, 4, 5, 6, 7
**Version:** v1.1.0
**Status:** Modern UI overhaul complete + GEO action plan + all core features

**Notes:**
- Restart Claude Code session at 40-50% context
- This PRD is the source of truth across sessions
