# SimCheck Roadmap

Single source for what's next. Priorities are proposals; reorder freely.
Shipped work lives in PRD.md (features 1-14) and the git history.

**Now:** v2.0.0 on `main` (2026-10-01): QRG page rating, AI access, citation
probes, site audit, tabbed UI, hosted login gate.

---

## Now: prove the ratings

| # | Item | Why | Size |
|---|---|---|---|
| 1 | **Rate the golden set** (Marc, `eval/PLAYBOOK.md`) | Nothing below is trustworthy for clients until ratings agree with a human ±1 level 80%+ of the time | 1 hr |
| 2 | **Pick the default rating mode** from the eval (Jev vs hybrid) | Claude changes 9/13 bands; unknown whether it's more right | S |
| 3 | **Tune rubric/overrides** if eval shows bias or misses | e.g. homepages consistently over-rated | S-M |

## Next: explain *why* AI doesn't cite you

| # | Item | Why | Size |
|---|---|---|---|
| 4 | **Citation gap**: rate the pages AI cited instead (snapshot + Jev + Content Match) and show what they have that this page lacks | The closest real "rater for LLMs": learned from what engines actually cite | M |
| 5 | **Source-type question** (earned / owned / social / UGC / reference) | AI search favors earned media (arXiv 2509.08919); owned pages need "get covered" advice, not just edits | S |
| 6 | **Probe history**: save probe runs to compare citation rate before/after edits | The "re-run in two weeks" promise in the client report needs a baseline | M |
| 7 | **More probe engines**: GPT web search (works, sparse citations); Claude (OpenRouter web plugin returned none; try server-side search tool) | Perplexity alone is one engine's view | S-M |
| 7a | **Brand accuracy** (Seer's Brand Canon): define 50+ brand facts → generate direct / indirect / comparative prompts → ask several LLMs → Jev checks each answer against each fact (yes/no with probability) → accuracy % over time, with cited sources for every miss | Seer: fix how LLMs describe the brand before chasing category visibility. Jev makes checking 100+ answers nearly free | L |
| 7b | **Probe query ladder**: organize probe questions by Seer's 4 stages (branded → branded + attribute → long-tail non-branded → non-branded) and report citation rate per stage | Shows where visibility breaks down, not just one overall number | S |
| 7c | **Local MCP server** (stdio, `python -m simcheck.mcp`): tools for analyze_page, rate_page, check_ai_access, content_match, run_probes, explain_rating, audit_site. Uses the local key file; paid tools state cost and respect the daily caps; page content returned as structured signals and labeled excerpts, never raw HTML (prompt injection) | Drive SimCheck from Claude Code or any MCP client: "audit this site, then draft fixes for the worst pages" | S-M |

## Later: make it client-ready

| # | Item | Why | Size |
|---|---|---|---|
| 8 | Agency branding on the shareable report (logo, name, colors) | Deferred by choice; needed before sending to clients | S |
| 9 | Google Sheet / Doc export (Drive OAuth, token outside repo) | Where client deliverables actually live | M |
| 10 | Site Audit: click a row to open that page's full report | Grid → detail is the natural drill-down | S |
| 11 | Compare two URLs side by side (yours vs a competitor) | Common client question; partly covered by #4 | M |
| 12 | Persist analyses (local JSON/SQLite) with simple history | Re-scoring drafts and tracking clients over time | M |
| 13 | Per-person access codes when the tester group grows (already supported) | Shared password is fine for light testing only | XS |

## Content Match (v1) carry-overs

| # | Item | Size |
|---|---|---|
| 14 | Compare two drafts against the same query | M |
| 15 | **Query fan-out coverage**: expand the target query into the sub-queries AI Mode would run (iPullRank) and score Content Match against each | M |
| 16 | Similarity histogram; configurable chunk size; custom thresholds in UI | S each |

## Platform

| # | Item | When |
|---|---|---|
| 17 | Move hosting to Hugging Face Spaces or Render | If Streamlit Cloud runs out of memory or sleeps too often |
| 18 | Remote API / remote MCP (separate FastAPI service; per-client tokens; reuse access.py limits) | When other people's apps need it, after #7c |
| 19 | Lighter embeddings (ONNX via fastembed, same bge model, no torch) | If hosting size or cold starts hurt |
| 20 | Move off Streamlit (FastAPI + front end) | Only if the UI limits start costing real use |

## Parked

- Automatic content rewrites: the tool diagnoses; editors write.
- Full-site crawls: sampling answers the question at a fraction of the cost.
- llms.txt as a scored factor: low impact until AI engines use it.

## References

- Google Search Quality Rater Guidelines, Sept 2025 edition (`docs/reference/`, gitignored).
- Aggarwal et al., *GEO: Generative Engine Optimization* (Princeton, KDD 2024): citations, statistics, and quotations raise AI visibility.
- *Generative Engine Optimization: How to Dominate AI Search*, [arXiv 2509.08919](https://arxiv.org/abs/2509.08919): AI search strongly favors earned media over brand-owned content (→ #5).
- Alisa Scharf (Seer Interactive), [Stop Chasing AI Rankings Before You Fix How LLMs See Your Brand](https://www.seerinteractive.com/insights/stop-chasing-ai-rankings-before-you-fix-how-llms-see-your-brand), May 2026: Brand Canon, accuracy %, 4-stage query ladder (→ #7a, #7b).
- iPullRank, [AI Search](https://ipullrank.com/ai-search): readiness model; query fan-out, crawler access, authority signals (→ #15). Their AI Search Manual is the deeper source.
- LangChain, [Building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev): typed classifier first, LLM only where needed.
