#!/usr/bin/env python3
"""Summarize WikiText-2 compensated PPL results.

This script only reads JSON artifacts produced by
run_compensated_ppl_downstream_gpu01234567.sh. It does not evaluate models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


METHODS: List[Tuple[str, str]] = [
    ("full", "Full"),
    ("static_ends_heavy", "Static ends_heavy"),
    ("static_best_on_val", "Static best-on-val"),
    ("pudding", "PuDDing-style"),
    ("ig", "IG-style"),
    ("layerwise", "layerwise_hidden_router"),
    ("raw_final", "Raw-SetBCE final epoch"),
    ("raw_best_val", "Raw-SetBCE best-on-val"),
    ("opal_final", "OPAL-SetBCE final epoch"),
    ("opal_best_val", "OPAL-SetBCE best-on-val"),
]


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace(",", " ").split() if x.strip()]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "pending"
    return f"{value:.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "pending"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if finite(v)]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def candidate_collapse(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return "pending"
    dist = payload.get("selected_candidate_distribution") or {}
    if not dist:
        return ""
    if len(dist) == 1:
        key, value = next(iter(dist.items()))
        return f"{key} ({value})"
    return ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))


def row_from_payload(
    label: str,
    payload: Optional[Dict[str, Any]],
    full_payload: Optional[Dict[str, Any]],
    artifact: Path,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": label,
        "artifact": str(artifact),
        "status": "ok" if payload else "missing",
    }
    if not payload:
        return row
    full_nll = float(full_payload["nll"]) if full_payload and finite(full_payload.get("nll")) else None
    full_ppl = float(full_payload["ppl"]) if full_payload and finite(full_payload.get("ppl")) else None
    nll = float(payload["nll"])
    ppl = float(payload["ppl"])
    row.update(
        {
            "nll": nll,
            "ppl": ppl,
            "delta_nll": nll - full_nll if full_nll is not None else None,
            "delta_ppl": ppl - full_ppl if full_ppl is not None else None,
            "unique_masks": payload.get("unique_masks"),
            "exact_skip_count_rate": payload.get("exact_skip_count_rate"),
            "average_kept_layers": payload.get("average_kept_layers"),
            "eval_tokens": payload.get("eval_tokens"),
            "compensation_runtime_calls": payload.get("compensation_runtime_calls"),
            "compensation_runtime_total_sec": payload.get("compensation_runtime_total_sec"),
            "selected_candidate_distribution": payload.get("selected_candidate_distribution", {}),
        }
    )
    return row


def collect(args: argparse.Namespace, seed: int) -> Dict[str, Dict[str, Any]]:
    metric_dir = Path(args.output_root) / "metrics" / f"seed{seed}"
    full_payload = load_json(metric_dir / "full.json")
    rows: Dict[str, Dict[str, Any]] = {}
    for slug, label in METHODS:
        path = metric_dir / f"{slug}.json"
        rows[label] = row_from_payload(label, load_json(path), full_payload, path)
    return rows


def wins(rows: Dict[str, Dict[str, Any]], lhs: str, rhs: str) -> Optional[bool]:
    left = rows.get(lhs, {})
    right = rows.get(rhs, {})
    if not finite(left.get("ppl")) or not finite(right.get("ppl")):
        return None
    return float(left["ppl"]) < float(right["ppl"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", default="42,13,3407")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--router_prefix_tokens", type=int, default=256)
    parser.add_argument("--label_samples", type=int, default=2000)
    parser.add_argument("--eval_windows", type=int, default=512)
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--skip_count", type=int, default=7)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--compensation_mode", default="ghost")
    parser.add_argument("--compensation_rank", type=int, default=64)
    parser.add_argument("--compensation_static_gate", type=float, default=1.0)
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    by_seed = {seed: collect(args, seed) for seed in seeds}

    lines = [
        "# WikiText-2 Compensated PPL Results",
        "",
        f"Last updated: {args.date}",
        "",
        "## Setup",
        "",
        f"- model: `{args.model}`",
        f"- seeds: `{seeds}`",
        f"- seq_len / router_prefix_tokens: `{args.seq_len}` / `{args.router_prefix_tokens}`",
        f"- label_samples / eval_windows: `{args.label_samples}` / `{args.eval_windows}`",
        f"- skip_rate / skip_count: `{args.skip_rate}` / `{args.skip_count}`",
        f"- protected_head / protected_tail: `{args.protected_head}` / `{args.protected_tail}`",
        f"- compensation: `{args.compensation_mode}` rank `{args.compensation_rank}` gate `{args.compensation_static_gate}`",
        "- checkpoint selection: Raw/OPAL best-on-val checkpoints are the existing no-comp WikiText validation selections; compensation is applied at evaluation time to every skip method.",
        "",
    ]

    for seed in seeds:
        rows = by_seed[seed]
        lines += [
            f"## Seed {seed}",
            "",
            "| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact_skip_count_rate | average_kept_layers | notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, label in METHODS:
            row = rows[label]
            note = ""
            if label in {"PuDDing-style", "IG-style"}:
                note = candidate_collapse(row)
            if row.get("status") != "ok":
                note = f"missing: `{row.get('artifact')}`"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        fmt(row.get("nll")),
                        fmt(row.get("ppl")),
                        fmt(row.get("delta_nll")),
                        fmt(row.get("delta_ppl")),
                        fmt_int(row.get("unique_masks")),
                        fmt(row.get("exact_skip_count_rate"), 3),
                        fmt(row.get("average_kept_layers"), 2),
                        note,
                    ]
                )
                + " |"
            )
        lines.append("")

    lines += [
        "## Three-Seed Mean/Std",
        "",
        "| method | NLL mean | NLL std | PPL mean | PPL std | Delta_NLL mean | Delta_PPL mean | unique_masks mean | exact_skip_count_rate mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mean_rows: Dict[str, Dict[str, Optional[float]]] = {}
    for _, label in METHODS:
        rows = [by_seed[seed][label] for seed in seeds]
        nll_mean, nll_std = mean_std(row.get("nll") for row in rows)
        ppl_mean, ppl_std = mean_std(row.get("ppl") for row in rows)
        delta_nll_mean, _ = mean_std(row.get("delta_nll") for row in rows)
        delta_ppl_mean, _ = mean_std(row.get("delta_ppl") for row in rows)
        unique_mean, _ = mean_std(row.get("unique_masks") for row in rows)
        exact_mean, _ = mean_std(row.get("exact_skip_count_rate") for row in rows)
        mean_rows[label] = {
            "nll": nll_mean,
            "ppl": ppl_mean,
            "delta_nll": delta_nll_mean,
            "delta_ppl": delta_ppl_mean,
            "unique_masks": unique_mean,
            "exact_skip_count_rate": exact_mean,
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(nll_mean),
                    fmt(nll_std),
                    fmt(ppl_mean),
                    fmt(ppl_std),
                    fmt(delta_nll_mean),
                    fmt(delta_ppl_mean),
                    fmt(unique_mean),
                    fmt(exact_mean, 3),
                ]
            )
            + " |"
        )

    opal = mean_rows.get("OPAL-SetBCE best-on-val", {}).get("ppl")
    lines += ["", "## Required Judgments", ""]
    for baseline in [
        "Raw-SetBCE best-on-val",
        "layerwise_hidden_router",
        "PuDDing-style",
        "IG-style",
        "Static best-on-val",
        "Static ends_heavy",
    ]:
        base = mean_rows.get(baseline, {}).get("ppl")
        verdict = bool(opal < base) if finite(opal) and finite(base) else None
        lines.append(f"- OPAL best-on-val beats `{baseline}` by three-seed mean PPL: `{verdict}`")

    opal_unique = [by_seed[seed]["OPAL-SetBCE best-on-val"].get("unique_masks") for seed in seeds]
    lines.append(f"- OPAL best-on-val unique_masks by seed: `{[round(float(v), 4) for v in opal_unique if finite(v)]}`")
    if any(finite(v) and float(v) <= 2 for v in opal_unique):
        lines.append("- Caveat: OPAL best-on-val remains static-like on at least one seed under compensation.")
    for method in ("PuDDing-style", "IG-style"):
        dists = []
        for seed in seeds:
            row = by_seed[seed][method]
            dists.append((seed, row.get("selected_candidate_distribution", {})))
        collapsed = all(isinstance(dist, dict) and len(dist) == 1 and "ends_heavy" in dist for _, dist in dists)
        lines.append(f"- {method} candidate distributions by seed: `{dists}`")
        lines.append(f"- {method} degenerates to ends_heavy only: `{collapsed}`")

    missing = []
    for seed, rows in by_seed.items():
        for label, row in rows.items():
            if row.get("status") != "ok":
                missing.append(f"seed{seed} {label}: {row.get('artifact')}")
    if missing:
        lines += ["", "## Missing Artifacts", ""]
        lines.extend(f"- `{item}`" for item in missing)

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
