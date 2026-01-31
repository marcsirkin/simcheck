# SimCheck

Local query-to-document cosine similarity analyzer for LLM visibility optimization.

SimCheck helps you understand how well your content semantically aligns with target concepts. It chunks documents, computes embeddings locally, and provides detailed diagnostics to identify where content drifts off-topic.

## Features

- **Local-only**: All processing happens on your machine using sentence-transformers
- **Chunk-level analysis**: See exactly which parts of your document align (or don't) with your concept
- **Concept Coverage Score (CCS)**: A weighted 0-100 score measuring how thoroughly your document expresses the target concept
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

# Run comparison
result = compare_query_to_document("machine learning", document_text)

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
```

## How It Works

1. **Chunking**: Documents are split into ~150 token chunks at sentence boundaries
2. **Embedding**: Each chunk (and the query) is embedded using `all-MiniLM-L6-v2`
3. **Similarity**: Cosine similarity is computed between the query and each chunk
4. **Scoring**: Chunks are bucketed by similarity:
   - Strong (≥0.80): Full alignment
   - Moderate (0.65-0.80): Good alignment
   - Weak (0.45-0.65): Partial alignment
   - Off-topic (<0.45): No alignment

### Concept Coverage Score (CCS)

CCS is a weighted score (0-100) that answers: "How much of this document meaningfully expresses the target concept?"

| Bucket | Weight |
|--------|--------|
| Strong | 1.0 |
| Moderate | 0.6 |
| Weak | 0.2 |
| Off-topic | 0.0 |

Formula: `CCS = (sum of weighted chunks / total chunks) × 100`

## Testing

```bash
pytest simcheck/tests/ -v
```

194 tests covering chunking, embeddings, similarity, diagnostics, and CCS calculation.

## License

MIT
