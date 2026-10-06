"""Evaluate Content Match labels against hand-rated query/passage pairs.

The input CSV must contain: id, query, passage, label. Labels are one of
off_topic, weak, moderate, strong. Keep real client examples in
eval/data/content_match_golden.csv, which is gitignored.

This runner reports the current threshold accuracy and compares plain query
encoding with BGE's recommended retrieval instruction. It does not rewrite
production thresholds.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simcheck.core.embeddings import embed_text, embed_texts  # noqa: E402
from simcheck.core.models import interpret_similarity, thresholds_for_query  # noqa: E402
from simcheck.core.similarity import cosine_similarity_normalized  # noqa: E402


LABELS = ("off_topic", "weak", "moderate", "strong")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "query", "passage", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    for row in rows:
        row["label"] = row["label"].strip().lower()
        if row["label"] not in LABELS:
            raise ValueError(f"Row {row['id']!r} has unknown label {row['label']!r}")
        if not row["query"].strip() or not row["passage"].strip():
            raise ValueError(f"Row {row['id']!r} has an empty query or passage")
    return rows


def score_rows(rows: list[dict], instructed: bool) -> list[float]:
    queries = [
        (QUERY_INSTRUCTION + row["query"]) if instructed else row["query"]
        for row in rows
    ]
    query_vectors = [embed_text(query) for query in queries]
    passage_vectors = embed_texts([row["passage"] for row in rows])
    return [
        cosine_similarity_normalized(query_vector, passage_vector)
        for query_vector, passage_vector in zip(query_vectors, passage_vectors)
    ]


def report(rows: list[dict], scores: list[float], title: str) -> None:
    grouped = defaultdict(list)
    confusion = Counter()
    correct = 0
    for row, score in zip(rows, scores):
        expected = row["label"]
        predicted = interpret_similarity(score, thresholds_for_query(row["query"])).lower().replace("-", "_")
        grouped[expected].append(score)
        confusion[(expected, predicted)] += 1
        correct += expected == predicted

    print(f"\n{title}")
    print(f"  current-threshold accuracy: {correct}/{len(rows)} ({correct / len(rows):.0%})")
    for label in LABELS:
        values = grouped[label]
        if values:
            print(
                f"  {label:10} n={len(values):2} mean={sum(values) / len(values):.3f} "
                f"range={min(values):.3f}-{max(values):.3f}"
            )
    print("  confusion (expected -> predicted):")
    for (expected, predicted), count in sorted(confusion.items()):
        print(f"    {expected:10} -> {predicted:10}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).parent / "data" / "content_match_golden.csv",
    )
    args = parser.parse_args()
    try:
        rows = load_rows(args.golden)
    except (OSError, ValueError) as exc:
        print(f"Content Match evaluation error: {exc}", file=sys.stderr)
        return 1

    print(f"Content Match golden set: {len(rows)} labeled query/passage pairs")
    report(rows, score_rows(rows, instructed=False), "Plain query encoding (current production)")
    report(rows, score_rows(rows, instructed=True), "BGE retrieval-instruction encoding")
    print("\nNo thresholds were changed. Tune only after reviewing the labeled distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
