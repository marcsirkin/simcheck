# QRG golden set

Hand ratings used to measure how well the classifier agrees with a human
rater, and to tune `ESCALATION_CONFIDENCE` and the hard-gate thresholds.

Ratings live in `eval/data/qrg_golden.csv` (gitignored: client sites).

## How to rate

Rate each page the way a Google Quality Rater would (QRG Sept 2025 edition,
`docs/reference/qrg.txt`). Rate before looking at any SimCheck output for the
page, so the tool does not anchor you.

| Column | Values |
|---|---|
| `include` | `yes` / `no` (no = exclude from scoring; say why in notes) |
| `page_quality` | `Lowest`, `Lowest+`, `Low`, `Low+`, `Medium`, `Medium+`, `High`, `High+`, `Highest` (QRG's 9-point slider) |
| `ymyl` | `no`, `possibly`, `clearly` |
| `purpose` | `informational`, `commercial`, `transactional`, `navigational`, `other` |
| `trust` | `Lowest`, `Low`, `Medium`, `High`, `Highest` |
| `notes` | One line on what drove the rating |

## Balance

Agreement numbers are only meaningful if the set spans the scale. Aim for
at least 3 pages rated Low or below and at least 5 article-level pages
(not homepages).
