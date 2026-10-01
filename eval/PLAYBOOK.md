# Golden-set rating playbook

Goal: find out whether SimCheck's Page Quality ratings agree with a human
rater, and whether Claude's help (hybrid mode) is worth its cost. You rate,
the script compares.

Time: about an hour for ~20 pages. Cost: a few cents (new pages get one Jev
call and one Claude call each; everything is cached after that).

---

## Rules before you start

1. **Don't look at SimCheck's rating for any page on the list until you've
   rated it.** Seeing its answer first anchors yours, and then the test
   measures nothing.
2. **Rate the page, not the brand.** A great company can have a thin page.
3. **Go with your first considered judgment.** Spend 2-4 minutes per page.
   Agonizing between two adjacent levels is what the "+" positions are for.

---

## Step 1. Open the sheet (2 min)

Open `eval/data/qrg_golden.csv` in VS Code or a spreadsheet app. It already
lists 15 URLs; 13 are marked `include = yes`. The two marked `no` (blocked or
JavaScript-only) stay out.

If you use Excel or Numbers, save as **CSV (UTF-8)**, not .xlsx.

## Step 2. Add pages so the test can tell good from bad (15 min)

Today the list is all homepages, mostly clients, probably all rating Medium
to High. A test where everything is "High" can't tell good judgment from
lazy judgment. Add about 8 rows, each with `include = yes`:

- **3+ weak pages.** Pages you'd honestly rate Low or Lowest, for example:
  - a thin "best X 2026" affiliate roundup with no evidence of testing
  - an obviously AI-churned article (generic, repetitive, no author)
  - an abandoned blog post with broken promises, outdated info, and ads
  - a health or money claim page with no author and no About page
- **5+ article pages, not homepages.** For example a Walk West blog post, a
  Semrush guide, a sirkin.com essay, and a news article or two.

Find candidates by searching normally. Before you add one, open it to make
sure it loads, isn't a login wall, and has real text on the page.

## Step 3. Rate each page (40 min)

Open the page. Skim the main content, then scroll to see who wrote it, the
date, the About/Contact pages, and the ads. Fill in one row:

| Column | Enter |
|---|---|
| `page_quality` | `Lowest` `Lowest+` `Low` `Low+` `Medium` `Medium+` `High` `High+` `Highest` |
| `ymyl` | `no` `possibly` `clearly` |
| `purpose` | `informational` `commercial` `transactional` `navigational` `other` |
| `trust` | `Lowest` `Low` `Medium` `High` `Highest` |
| `notes` | One line: what drove the rating |

### Page Quality cheat sheet (Google QRG, Sept 2025)

Start at Medium and move up or down.

| Level | Pick it when |
|---|---|
| **Lowest** | Deceptive, harmful, or scammy; no real main content; copied or auto-generated with no added value; YMYL with no way to tell who's responsible. **Any one** of these is enough. |
| **Low** | Little effort or originality; thin for its purpose; ads or affiliate links get in the way; title oversells; not enough expertise for the topic; no clear author or owner on a topic that needs one. |
| **Medium** | Does its job. Nothing wrong, nothing special. Or a mix of good and bad. |
| **High** | Real effort and skill; satisfying for its purpose; clear who's behind it and why they're credible; good reputation. |
| **Highest** | Exceptional effort or expertise; the recognized go-to source for this topic; very strong reputation. |

Use `+` when it sits between two levels (for example `Medium+` = better than
Medium, not quite High).

### YMYL

Could inaccurate information here seriously hurt someone's **health, money,
safety, or civic life**?
`clearly` = yes (medical advice, investing, legal, news about public
safety), `possibly` = depends on details (a supplement review, a car
buying guide), `no` = no (a wine brand homepage, a SaaS feature page).

YMYL raises the bar: the same anonymous page can be Medium on wine and Low
on medication.

### Trust

The most important part of E-E-A-T. Is the page accurate and honest, and is it
clear who's responsible for it? Rate it on the same five levels.

### Example rows

```
https://example.com/best-air-fryers,yes,Low,no,commercial,Low,templated roundup no testing heavy affiliate
https://example.gov/high-blood-pressure,yes,Highest,clearly,informational,Highest,NIH source reviewed clear ownership
```

## Step 4. Run the comparison (2 min)

```bash
cd ~/projects/cosine-similiarity-tester
.venv/bin/python eval/run_eval.py
```

New pages get fetched and classified once (a few seconds each). Re-runs use
the cache and cost nothing. A page that fails to fetch is skipped with a
message; set its `include` to `no` and add its reason in `notes`.

## Step 5. Read the result (5 min)

The script prints two sections.

**"How much does Claude change Jev's rating?"** How often Claude disagrees.
It doesn't need your ratings and isn't the verdict.

**"Agreement with human ratings"** is the verdict. One row per setup:

| Column | Means | Good |
|---|---|---|
| `exact` | Same 9-point position as you | Nice to have |
| `±half` | Within one step (High vs High+) | 60%+ |
| `±1lvl` | Within one full level (High vs Medium) | **80%+: the bar for client use** |
| `MAE` | Average distance from you, in levels | Under 0.75 |
| `bias` | + rates higher than you, − lower | Near 0 |
| `YMYL`, `purpose` | Agreement on those labels | 80%+ |

To see page-by-page disagreements, after you've rated:

```bash
.venv/bin/python eval/run_eval.py --show-pages
```

## Step 6. Decide (with Claude Code)

| Result | Decision |
|---|---|
| Jev only ≥ 80% ±1lvl and close to the hybrid rows | Keep Jev only (fast, nearly free) |
| A hybrid row beats Jev by 10+ points on ±1lvl | Switch default to hybrid at that threshold |
| Everything under 80% | Don't use ratings in client reports yet. Look at `--show-pages` for patterns (for example always too high on homepages) and tune the rubric |
| Large consistent bias | Rubric wording or overrides need tuning, not the backend |

Paste the output into a Claude Code session in this repo and say "golden set
results". The change is one line (`DEFAULT_MODE` in
`simcheck/quality/classifier.py`) or rubric and override tuning, re-run to
confirm.

## Later rounds

- Add pages over time, especially ones where SimCheck surprised you.
- After any rubric change, `RUBRIC_VERSION` bumps and the script re-classifies
  automatically. Your ratings stay valid.
- Keep the sheet local. It's gitignored because it holds client sites.
