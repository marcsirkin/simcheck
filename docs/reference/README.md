# Reference material

`qrg.pdf` / `qrg.txt` — Google Search Quality Rater Guidelines (182 pp).
Not committed (Google copyright, 9 MB). Re-fetch:

```bash
curl -sSL -o docs/reference/qrg.pdf https://guidelines.raterhub.com/searchqualityevaluatorguidelines.pdf
pdftotext -layout docs/reference/qrg.pdf docs/reference/qrg.txt   # brew install poppler
```

The rubric in `simcheck/quality/rubric.py` is derived from this document;
`RUBRIC_VERSION` records which QRG edition it tracks.
