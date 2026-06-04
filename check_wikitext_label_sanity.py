#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def finite_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def stat(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p1": percentile(values, 1),
        "p5": percentile(values, 5),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Fail-fast sanity check for WikiText greedy skip-set labels.")
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--min_rows", type=int, default=1)
    parser.add_argument("--max_full_nll_mean", type=float, default=5.0)
    parser.add_argument("--max_full_ppl_median", type=float, default=200.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    path = Path(args.label_file)
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty label file: {path}")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < int(args.min_rows):
        raise SystemExit(f"Label file has {len(rows)} rows, expected at least {args.min_rows}: {path}")

    nll_full = [finite_float(row.get("NLL_full")) for row in rows]
    ppl_full = [finite_float(row.get("PPL_full")) for row in rows]
    nll_full = [value for value in nll_full if value is not None]
    ppl_full = [value for value in ppl_full if value is not None]
    if len(nll_full) != len(rows) or len(ppl_full) != len(rows):
        raise SystemExit(
            f"Refusing label file without finite NLL_full/PPL_full for every row: "
            f"rows={len(rows)} nll_full={len(nll_full)} ppl_full={len(ppl_full)} path={path}"
        )

    nll_stats = stat(nll_full)
    ppl_stats = stat(ppl_full)
    payload = {
        "path": str(path),
        "rows": len(rows),
        "NLL_full": nll_stats,
        "PPL_full": ppl_stats,
        "thresholds": {
            "max_full_nll_mean": float(args.max_full_nll_mean),
            "max_full_ppl_median": float(args.max_full_ppl_median),
        },
    }
    if not args.quiet:
        print(json.dumps(payload, indent=2))

    bad = []
    if float(nll_stats["mean"]) > float(args.max_full_nll_mean):
        bad.append(f"NLL_full_mean={nll_stats['mean']:.6g}>{args.max_full_nll_mean:.6g}")
    if float(ppl_stats["median"]) > float(args.max_full_ppl_median):
        bad.append(f"PPL_full_median={ppl_stats['median']:.6g}>{args.max_full_ppl_median:.6g}")
    if bad:
        raise SystemExit(f"Refusing poisoned WikiText label file: {', '.join(bad)} path={path}")


if __name__ == "__main__":
    main()
