#!/usr/bin/env python3
"""Summarize WikiText-2 shared compensation fairness results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


METHODS = [
    ("static_best_on_val", "Static best-on-val"),
    ("pudding", "PuDDing-style"),
    ("ig", "IG-style"),
    ("raw_best_val", "Raw-SetBCE best-on-val"),
    ("layerwise", "layerwise_hidden_router"),
    ("opal_best_val", "OPAL-SetBCE best-on-val"),
]


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


def row(
    method: str,
    no_comp: Optional[Dict[str, Any]],
    comp: Optional[Dict[str, Any]],
    full: Optional[Dict[str, Any]],
    adapter_meta: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": method}
    if no_comp:
        out["no_comp_nll"] = no_comp.get("nll")
        out["no_comp_ppl"] = no_comp.get("ppl")
        out["unique_masks"] = no_comp.get("unique_masks")
        out["exact_skip_count_rate"] = no_comp.get("exact_skip_count_rate")
        out["average_skipped_layers"] = no_comp.get("average_skipped_layers")
    if comp:
        out["comp_nll"] = comp.get("nll")
        out["comp_ppl"] = comp.get("ppl")
        out["compensation_adapter_param_count"] = comp.get("compensation_adapter_param_count") or adapter_meta.get("adapter_param_count")
        out["compensation_runtime_calls"] = comp.get("compensation_runtime_calls")
    if full and finite(full.get("nll")):
        full_nll = float(full["nll"])
        if finite(out.get("no_comp_nll")):
            out["no_comp_delta_nll"] = float(out["no_comp_nll"]) - full_nll
        if finite(out.get("comp_nll")):
            out["comp_delta_nll"] = float(out["comp_nll"]) - full_nll
    if full and finite(full.get("ppl")):
        full_ppl = float(full["ppl"])
        if finite(out.get("no_comp_ppl")):
            out["no_comp_delta_ppl"] = float(out["no_comp_ppl"]) - full_ppl
        if finite(out.get("comp_ppl")):
            out["comp_delta_ppl"] = float(out["comp_ppl"]) - full_ppl
    if finite(out.get("no_comp_ppl")) and finite(out.get("comp_ppl")):
        out["gain"] = float(out["no_comp_ppl"]) - float(out["comp_ppl"])
    return out


def replace_section(text: str, marker: str, section: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = f"{start}\n{section.rstrip()}\n{end}"
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        return before.rstrip() + "\n\n" + block + "\n" + after
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--related_md", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--router_prefix_tokens", type=int, default=256)
    parser.add_argument("--train_windows", type=int, default=2000)
    parser.add_argument("--eval_windows", type=int, default=512)
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--skip_count", type=int, default=7)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    seed_dir = Path(args.output_root) / "metrics" / f"seed{args.seed}"
    full = load_json(seed_dir / "full_no_comp.json")
    adapter_metrics = load_json(Path(args.output_root) / "adapter" / f"seed{args.seed}" / "training_metrics.json") or {}
    adapter_meta = adapter_metrics.get("metadata") or {}

    rows: List[Dict[str, Any]] = []
    for slug, label in METHODS:
        rows.append(
            row(
                label,
                load_json(seed_dir / f"{slug}_no_comp.json"),
                load_json(seed_dir / f"{slug}_same_comp.json"),
                full,
                adapter_meta,
            )
        )

    lines = [
        "# Shared Compensation Fairness Results",
        "",
        f"Last updated: {args.date}",
        "",
        "## Setup",
        "",
        f"- model: `{args.model}`",
        f"- seed: `{args.seed}`",
        f"- seq_len / router_prefix_tokens: `{args.seq_len}` / `{args.router_prefix_tokens}`",
        f"- train_windows / eval_windows: `{args.train_windows}` / `{args.eval_windows}`",
        f"- skip_rate / skip_count: `{args.skip_rate}` / `{args.skip_count}`",
        f"- protected_head / protected_tail: `{args.protected_head}` / `{args.protected_tail}`",
        f"- shared adapter: per-layer low-rank residual `h_out = h_in + B_l A_l RMSNorm(h_in)`, rank `{args.rank}`",
        f"- adapter scope: `{adapter_meta.get('scope', 'single_shared_adapter_across_methods')}`",
        f"- adapter params: `{adapter_meta.get('adapter_param_count', 'pending')}`",
        f"- base LLM frozen: `{adapter_meta.get('base_model_frozen', True)}`",
        f"- training objective: `KL(full logits || skipped+adapter logits)` on WikiText-2 train windows",
        "",
        "## WikiText-2 PPL",
        "",
        "| method | no-comp PPL | +same-comp PPL | gain | no-comp Delta_NLL | +comp Delta_NLL | no-comp Delta_PPL | +comp Delta_PPL | adapter params | compensated layer count | exact_skip_count_rate | unique_masks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["method"]),
                    fmt(item.get("no_comp_ppl")),
                    fmt(item.get("comp_ppl")),
                    fmt(item.get("gain")),
                    fmt(item.get("no_comp_delta_nll")),
                    fmt(item.get("comp_delta_nll")),
                    fmt(item.get("no_comp_delta_ppl")),
                    fmt(item.get("comp_delta_ppl")),
                    str(item.get("compensation_adapter_param_count", adapter_meta.get("adapter_param_count", "pending"))),
                    fmt(item.get("average_skipped_layers"), 2),
                    fmt(item.get("exact_skip_count_rate"), 3),
                    str(item.get("unique_masks", "pending")),
                ]
            )
            + " |"
        )

    best_comp = min(
        [item for item in rows if finite(item.get("comp_ppl"))],
        key=lambda item: float(item["comp_ppl"]),
        default=None,
    )
    opal = next((item for item in rows if item["method"] == "OPAL-SetBCE best-on-val"), None)
    lines += ["", "## Required Judgments", ""]
    if best_comp:
        lines.append(f"- Best +same-comp PPL method: `{best_comp['method']}` with PPL `{fmt(best_comp.get('comp_ppl'))}`")
    if opal and finite(opal.get("comp_ppl")):
        for item in rows:
            if item is opal or not finite(item.get("comp_ppl")):
                continue
            lines.append(
                f"- OPAL +same-comp beats `{item['method']}` by PPL: "
                f"`{float(opal['comp_ppl']) < float(item['comp_ppl'])}`"
            )
        lines.append(f"- OPAL compensation gain: `{fmt(opal.get('gain'))}` PPL")
    if any(finite(item.get("gain")) and float(item["gain"]) < 0 for item in rows):
        lines.append("- Caveat: at least one method has negative compensation gain; report this directly.")

    missing = []
    for slug, label in METHODS:
        for suffix in ("no_comp", "same_comp"):
            path = seed_dir / f"{slug}_{suffix}.json"
            if not path.exists():
                missing.append(f"{label} {suffix}: {path}")
    if missing:
        lines += ["", "## Missing Artifacts", ""]
        lines.extend(f"- `{item}`" for item in missing)

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")

    if args.related_md:
        related_path = Path(args.related_md)
        existing = related_path.read_text(encoding="utf-8") if related_path.exists() else ""
        section = "\n".join(lines[2:])
        related_path.write_text(replace_section(existing, "SHARED_COMP_FAIRNESS", section), encoding="utf-8")
        print(f"Updated {related_path}")


if __name__ == "__main__":
    main()
