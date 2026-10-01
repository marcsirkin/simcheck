# SimCheck

How Google would rate a page, and whether AI search cites it.

Give SimCheck a URL. It rates the page against Google's Search Quality Rater Guidelines, checks whether AI crawlers can reach it, tests whether AI answer engines actually cite it, and measures how closely the text covers the query you care about. The Report tab turns all of that into one finding and a short list of what to fix first.

## What it does

| Tab | Answers | How |
|---|---|---|
| **Report** | What's the verdict, and what do I fix first? | Generated headline finding, three figures, summary, ranked fixes. Exports a shareable client report (HTML) and full JSON. |
| **Page Quality** | How would a Google Quality Rater score this page? | 15 rubric questions from the QRG (Sept 2025 edition) answered by Jev, TypeSafe's typed classifier. Band on Google's 9-point scale plus a 0-100 score, E-E-A-T breakdown, YMYL, Needs Met. "Explain this rating" asks Claude for quoted evidence. |
| **LLM Visibility** | Can AI reach it? Does AI cite it? | robots.txt rules for 9 AI crawlers (search vs training), noindex/nosnippet, JavaScript rendering, schema, llms.txt. Click-to-run citation probes ask Perplexity real questions and show who got cited instead. |
| **Content Match** | Does the text cover the query? | The original SimCheck: local embeddings, Concept Coverage Score, SimScore, drift map, GEO action plan. Paste a draft here to re-score edits. |
| **Site Audit** | How does the whole site rate? | Samples pages from the sitemap across site sections, rates each one, sortable grid, CSV export. |

## Setup

```bash
git clone https://github.com/marcsirkin/simcheck.git
cd simcheck
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### API keys

Content Match runs fully offline. The other tabs need keys:

| Key | Used for | Without it |
|---|---|---|
| `TYPESAFE_API_KEY` | Page Quality and Site Audit ratings (Jev) | No ratings; access checks still run |
| `OPENROUTER_API_KEY` | "Explain this rating" (Claude) and citation probes (Perplexity) | Those buttons are disabled |

Keys live **outside the repo** in `~/.config/simcheck/.env`:

```bash
mkdir -p ~/.config/simcheck && chmod 700 ~/.config/simcheck
cp .env.example ~/.config/simcheck/.env
chmod 600 ~/.config/simcheck/.env   # SimCheck refuses to load a file others can read
# then paste your keys into it
```

Override the location with `SIMCHECK_ENV_FILE=/path/to/file`. Shell environment variables take precedence over the file. Jev is not available through OpenRouter (only `typesafe/jev-router`, a model router), so it needs its own TypeSafe key.

Set a credit limit on the OpenRouter key in its dashboard.

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501, paste a URL, optionally add the question a searcher would ask, and press **Analyze**.

## Costs

| Action | Backend | Cost |
|---|---|---|
| Analyze (rating) | Jev | about 0.3 s per page, fractions of a cent |
| Site Audit | Jev, one call per page | same per page; no Claude |
| Explain this rating | Claude via OpenRouter | about $0.01 |
| Citation probes | Perplexity Sonar via OpenRouter | about $0.005 per question |

Probes and explanations run only when you click. Nothing paid runs on Analyze except the Jev rating.

## How the rating works

1. **Snapshot**: fetch the page (http/https only, every redirect hop checked against private addresses, 5 MB cap) and extract title, schema, author, dates, main content, citations, statistics, About/Contact/policy links, and ad/affiliate signals. Bot-protected sites are reported as blocked, and you can paste the page HTML instead.
2. **Rubric**: Jev answers page purpose, YMYL, main-content quality, the four E-E-A-T dimensions, reputation, ad obstruction, deception, scaled low-effort content, and overall Page Quality in one parallel call, with probabilities.
3. **Overrides from the QRG** take precedence over the model: deceptive or harmful goes to Lowest; mass-produced content, ad-dominated pages, and anonymous YMYL pages are capped at Low.
4. **Score**: the band is the holistic Page Quality answer on Google's 9-point slider. The 0-100 score blends it with E-E-A-T (Trust weighted highest); E-E-A-T can lift it at most one level.
5. Pages with almost no server-rendered text are returned **Unrated**: that's a crawlability finding, not a quality judgment.

Rating mode defaults to Jev only. A hybrid mode (Jev first, Claude re-asks low-confidence questions) and a Claude-only mode exist in `simcheck.quality.classifier.make_classifier`; the default changes only if the golden-set evaluation shows Claude agrees with human raters more.

### Evaluating against human ratings

`eval/` measures agreement with hand ratings:

```bash
# Rate pages in eval/data/qrg_golden.csv (gitignored; see eval/README.md), then:
python eval/run_eval.py
```

Each backend is called once per page and cached, so threshold sweeps replay for free. The report shows exact, within-half-step, and within-one-level agreement, bias, and YMYL/purpose/trust agreement for Jev, Claude, and hybrid at several thresholds.

## Content Match reference

1. **Chunking**: flat (~150-token chunks at sentence boundaries) or hierarchical (H2 → MACRO, H3 → MICRO, paragraphs → ATOMIC) via `FLAT`, `MARKDOWN`, `HTML`, or `AUTO`.
2. **Embedding**: `BAAI/bge-base-en-v1.5`, locally.
3. **Similarity**: cosine similarity between the query and each chunk, bucketed Strong (≥0.80), Moderate (0.65-0.80), Weak (0.45-0.65), Off-topic (<0.45). Queries of 1-2 words use lower calibrated bands (0.72 / 0.60 / 0.42) because bare keywords score systematically lower.
4. **Concept Coverage Score**: `CCS = (sum of weighted chunks / total chunks) × 100` with weights 1.0 / 0.6 / 0.2 / 0.0. CCS is relative: compare drafts for the same topic, not unrelated pages.
5. **SimScore**: 0-100 composite of coverage (50%), structure (20%), evidence (15%), and answerability (15%).

### Python API

```python
from simcheck.config import load_api_keys
from simcheck.quality.snapshot import snapshot_url
from simcheck.quality.ai_access import check_ai_access
from simcheck.quality.classifier import make_classifier
from simcheck.quality.page_quality import rate_page

snapshot = snapshot_url("https://example.com/guide")
access = check_ai_access(snapshot)
rating = rate_page(snapshot, make_classifier(load_api_keys()), access, query="what is dkim")
print(rating.band, rating.pq_score_rounded, rating.reasons)
```

```python
from simcheck import compare_query_to_document, create_diagnostic_report

result = compare_query_to_document("DKIM email authentication", document_text)
report = create_diagnostic_report(result)
print(report.coverage.score_rounded, report.coverage.interpretation)
```

## Security

- API keys are read only by `simcheck/config.py`, from a chmod-600 file outside the repo, and are never printed (only the last 4 characters).
- A gitleaks pre-commit hook blocks secrets: `git config core.hooksPath .githooks` (needs `brew install gitleaks`).
- Fetched page content is untrusted: it is HTML-escaped everywhere it renders, and passed to Claude as delimited data with instructions never to follow it.
- Client data stays local: `eval/data/`, `runs/`, and `exports/` are gitignored.

## Testing

```bash
pytest simcheck/tests/ -v
```

522 tests. No test calls the network: classifiers, OpenRouter, and HTTP are faked.

## License

MIT
