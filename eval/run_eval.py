"""
Evaluate Page Quality ratings against the hand-rated golden set.

Usage:
    python eval/run_eval.py                 # label-free comparison + metrics for rated rows
    python eval/run_eval.py --no-claude     # Jev only (no OpenRouter spend)
    python eval/run_eval.py --refresh       # re-fetch pages and re-classify
    python eval/run_eval.py --show-pages    # per-page bands (only AFTER you've rated)

Caches (eval/data/cache/, gitignored):
    snapshots/*.pkl   fetched PageSnapshots
    answers.jsonl     raw answers per (url, backend, rubric_version)

Each backend is called once per page; threshold sweeps replay the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simcheck.config import load_api_keys  # noqa: E402
from simcheck.quality.classifier import (  # noqa: E402
    ClaudeClassifier,
    ClassifierError,
    JevClassifier,
    build_state,
)
from simcheck.quality.evaluation import (  # noqa: E402
    CLAUDE_COST_PER_ESCALATED_PAGE,
    answers_from_json,
    answers_to_json,
    band_changes,
    compute_metrics,
    load_golden,
    rate_with_answers,
    read_cache,
    simulate_hybrid,
    write_cache,
)
from simcheck.quality.llm_client import OpenRouterClient  # noqa: E402
from simcheck.quality.rubric import RUBRIC_VERSION, questions_for  # noqa: E402
from simcheck.quality.snapshot import SnapshotError, snapshot_url  # noqa: E402


DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "cache"
SNAPSHOTS = CACHE / "snapshots"
ANSWERS = CACHE / "answers.jsonl"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
LIVE_THRESHOLD = 0.6


def _included_urls(golden_path: Path) -> list:
    import csv
    with open(golden_path, newline="") as f:
        return [r["url"].strip() for r in csv.DictReader(f)
                if (r.get("include") or "").strip().lower() == "yes"]


def get_snapshot(url: str, refresh: bool):
    """Snapshot from cache, or fetch and cache it. Pickles are our own local files."""
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".pkl")
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    snap = snapshot_url(url)
    path.write_bytes(pickle.dumps(snap))
    return snap


def get_answers(url: str, snap, backend, cache: dict, refresh: bool) -> dict:
    """Answers for one backend, from cache or a live call (then cached)."""
    key = (url, backend.name, RUBRIC_VERSION)
    if key in cache and not refresh:
        return answers_from_json(cache[key]["answers"])
    answers = backend.classify(build_state(snap), questions_for(None))
    record = {"url": url, "backend": backend.name, "rubric_version": RUBRIC_VERSION,
              "answers": answers_to_json(answers), "fetched_at": time.time()}
    write_cache(ANSWERS, record)
    cache[key] = record
    return answers


def _fmt(x, pct=True):
    if x is None:
        return "  n/a"
    return f"{x * 100:4.0f}%" if pct else f"{x:5.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DATA / "qrg_golden.csv")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude (no OpenRouter spend)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch and re-classify everything")
    parser.add_argument("--show-pages", action="store_true", help="Print per-page bands (after rating!)")
    args = parser.parse_args()

    keys = load_api_keys()
    if not keys.has_typesafe:
        print("TYPESAFE_API_KEY not configured.", file=sys.stderr)
        return 1
    jev = JevClassifier(api_key=keys.typesafe)
    claude = None
    if not args.no_claude:
        if not keys.has_openrouter:
            print("OPENROUTER_API_KEY not configured; running Jev only.", file=sys.stderr)
        else:
            claude = ClaudeClassifier(OpenRouterClient(api_key=keys.openrouter))

    cache = read_cache(ANSWERS)
    pages = []  # (url, snapshot, jev_answers, claude_answers|None)
    for url in _included_urls(args.golden):
        try:
            snap = get_snapshot(url, args.refresh)
            ja = get_answers(url, snap, jev, cache, args.refresh)
            ca = get_answers(url, snap, claude, cache, args.refresh) if claude else None
        except (SnapshotError, ClassifierError) as e:
            print(f"  skip {url}: {e}", file=sys.stderr)
            continue
        pages.append((url, snap, ja, ca))

    print(f"Rubric {RUBRIC_VERSION} | {len(pages)} pages\n")
    jev_ratings = [rate_with_answers(s, ja) for _, s, ja, _ in pages]

    # ---- Label-free: how much does Claude change things? ----
    if claude:
        claude_ratings = [rate_with_answers(s, ca) for _, s, _, ca in pages]
        hybrid_live = [rate_with_answers(s, simulate_hybrid(ja, ca, LIVE_THRESHOLD)[0]) for _, s, ja, ca in pages]
        print("How much does Claude change Jev's rating? (no labels needed)")
        for name, other in (("Claude only", claude_ratings), (f"Hybrid @{LIVE_THRESHOLD}", hybrid_live)):
            c = band_changes(jev_ratings, other)
            print(f"  {name:13} vs Jev: band changed on {c['slider_changed']}/{c['pages']} pages, "
                  f"by >=1 full level on {c['level_changed_1plus']}, mean |shift| {c['mean_abs_shift']:.2f}, "
                  f"mean shift {c['mean_shift']:+.2f}")
        print("\n  Escalation by threshold:")
        for t in THRESHOLDS:
            escalated_pages = sum(1 for _, _, ja, ca in pages if simulate_hybrid(ja, ca, t)[1])
            print(f"    {t:.1f}: {escalated_pages:2}/{len(pages)} pages escalate "
                  f"(~${escalated_pages * CLAUDE_COST_PER_ESCALATED_PAGE:.2f} per run of this set)")
        print()

    # ---- Labeled: agreement with the human rater ----
    labels = {g.url: g for g in load_golden(args.golden)}
    labeled = [(i, labels[url]) for i, (url, *_) in enumerate(pages) if url in labels]
    if not labeled:
        print("No rated rows in the golden CSV yet: fill in page_quality to get agreement metrics.")
        return 0

    rows = [("Jev only", [jev_ratings[i] for i, _ in labeled])]
    if claude:
        rows.append(("Claude only", [claude_ratings[i] for i, _ in labeled]))
        for t in THRESHOLDS:
            preds = []
            for i, _ in labeled:
                _, snap, ja, ca = pages[i]
                preds.append(rate_with_answers(snap, simulate_hybrid(ja, ca, t)[0]))
            rows.append((f"Hybrid @{t:.1f}", preds))

    print(f"Agreement with human ratings ({len(labeled)} rated pages)")
    print(f"  {'backend':14} {'exact':>6} {'±half':>6} {'±1lvl':>6} {'MAE':>6} {'bias':>6} {'YMYL':>6} {'purpose':>7} {'trustMAE':>8}")
    for name, preds in rows:
        m = compute_metrics([(g, p) for (_, g), p in zip(labeled, preds)])
        print(f"  {name:14} {_fmt(m.exact_slider)} {_fmt(m.within_half)} {_fmt(m.within_one)} "
              f"{_fmt(m.mae_levels, False)} {m.mean_bias:+6.2f} {_fmt(m.ymyl_agreement)} "
              f"{_fmt(m.purpose_agreement):>7} {_fmt(m.trust_mae, False):>8}")

    if args.show_pages:
        print("\nPer page (human vs Jev" + (" vs Claude" if claude else "") + ")")
        for i, g in labeled:
            line = f"  {pages[i][0]:40} human={g.level:.1f} jev={jev_ratings[i].band:8}"
            if claude:
                line += f" claude={claude_ratings[i].band:8}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
