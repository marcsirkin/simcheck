# Content Match calibration

This evaluation is separate from the Google QRG golden set. It tests whether
the chunk labels shown by Content Match agree with human judgments.

Create `eval/data/content_match_golden.csv` with:

```csv
id,query,passage,label
```

Use one passage-sized chunk per row. Labels are `off_topic`, `weak`,
`moderate`, or `strong`. Include several page purposes and both short and
searcher-shaped queries. Rate the rows before looking at SimCheck's score.

Run:

```bash
.venv/bin/python eval/run_content_match_eval.py
```

The runner reports the current threshold accuracy and the score distribution
for each label. It also compares current plain query encoding with BGE's
recommended retrieval instruction. It deliberately does not update production
thresholds; make that decision only after the distributions are large and
stable enough to justify it.
