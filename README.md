# SimCheck

Local query-to-document cosine similarity analyzer for LLM visibility optimization.

SimCheck helps you understand how well your content semantically aligns with target concepts. It chunks documents, computes embeddings locally, and provides detailed diagnostics to identify where content drifts off-topic.

## Features

- **Local-only**: All processing happens on your machine using sentence-transformers
- **Chunk-level analysis**: See exactly which parts of your document align (or don't) with your concept
- **Concept Coverage Score (CCS)**: A weighted 0-100 score measuring how thoroughly your document expresses the target concept
- **Hierarchical chunking**: Parse markdown/HTML structure to preserve document hierarchy (sections, subsections, paragraphs)
- **Improvement recommendations**: Actionable suggestions to boost your CCS, prioritized by impact
- **Interactive UI**: Streamlit playground for exploring results

## Installation

```bash
# Clone the repo
git clone https://github.com/marcsirkin/simcheck.git
cd simcheck

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. Enter a concept/query and paste your document to analyze.

### Python API

```python
from simcheck import compare_query_to_document, create_diagnostic_report
from simcheck.core.recommendations import generate_recommendations
from simcheck.core.chunker import ChunkingStrategy

# Run comparison (with optional hierarchical chunking)
result = compare_query_to_document(
    "machine learning",
    document_text,
    chunking_strategy=ChunkingStrategy.MARKDOWN,  # or FLAT, HTML, AUTO
)

print(f"Max similarity: {result.max_similarity:.2f}")
print(f"Avg similarity: {result.avg_similarity:.2f}")
print(f"Chunks analyzed: {result.chunk_count}")

# Get detailed diagnostics
report = create_diagnostic_report(result)

print(f"Concept Coverage Score: {report.coverage.score_rounded}/100")
print(f"Interpretation: {report.coverage.interpretation}")

# Find weak spots
for chunk in report.off_topic_chunks():
    print(f"Off-topic: {chunk.text_preview}")

# Get improvement recommendations
recs = generate_recommendations(report)
print(f"\nPotential CCS: {recs.potential_ccs:.0f} (+{recs.potential_ccs - recs.current_ccs:.0f})")

for rec in recs.high_priority():
    print(f"[HIGH] {rec.what}")
    print(f"  Fix: {rec.how}")
```

## How It Works

1. **Chunking**: Documents are split into semantic chunks
   - **Flat mode**: ~150 token chunks at sentence boundaries
   - **Hierarchical mode**: Preserves document structure (H2 → MACRO, H3 → MICRO, paragraphs → ATOMIC)
2. **Embedding**: Each chunk (and the query) is embedded using `BAAI/bge-base-en-v1.5`
3. **Similarity**: Cosine similarity is computed between the query and each chunk
4. **Scoring**: Chunks are bucketed by similarity:
   - Strong (≥0.80): Full alignment
   - Moderate (0.65-0.80): Good alignment
   - Weak (0.45-0.65): Partial alignment
   - Off-topic (<0.45): No alignment

### Chunking Strategies

| Strategy | Use Case |
|----------|----------|
| `FLAT` | Default, sentence-based chunking |
| `MARKDOWN` | Parse `##` and `###` headings |
| `HTML` | Parse `<h2>` and `<h3>` tags |
| `AUTO` | Auto-detect format |

### Concept Coverage Score (CCS)

CCS is a weighted score (0-100) that answers: "How much of this document meaningfully expresses the target concept?"

| Bucket | Weight |
|--------|--------|
| Strong | 1.0 |
| Moderate | 0.6 |
| Weak | 0.2 |
| Off-topic | 0.0 |

Formula: `CCS = (sum of weighted chunks / total chunks) × 100`

### Improvement Recommendations

SimCheck analyzes your diagnostic report and generates prioritized recommendations:

| Type | Trigger | Priority |
|------|---------|----------|
| Rewrite Off-Topic | Chunks < 0.45 | HIGH (if >15% off-topic) |
| Strengthen Weak | Chunks 0.45-0.65 | MEDIUM |
| Expand Strong | Use strong chunks as templates | LOW |
| Restructure Section | Section coverage < 40 | MEDIUM |
| Remove Dilution | Strong content diluted by off-topic | LOW |

Each recommendation includes:
- **What**: The problem identified
- **Why**: Impact on your CCS
- **How**: Actionable steps to fix
- **Affected chunks**: Specific content to address
- **Potential CCS**: Estimated score after fixes

## Testing

```bash
pytest simcheck/tests/ -v
```

269 tests covering chunking, embeddings, similarity, diagnostics, CCS, hierarchical parsing, and recommendations.

## License

MIT
