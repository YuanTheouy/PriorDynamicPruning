#!/usr/bin/env python3
"""Summarize WikiText-2 three-seed public LM sanity results.

This script only reads JSON artifacts produced by the WikiText-2 runners and
writes a Markdown report. It does not run model evaluation or train routers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


METHODS = [
    "Full",
    "Static ends_heavy",
    "Static best-on-val",
    "PuDDing-style",
    "IG-style",
    "layerwise_hidden_router",
    "Raw-SetBCE final epoch",
    "Raw-SetBCE best-on-val",
    "OPAL-SetBCE final epoch",
    "OPAL-SetBCE best-on-val",
]


def model_tag(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name.translate(str.maketrans({" ": "_", ".": "_", "/": "_", ":": "_"}))


def run_id(args: argparse.Namespace, seed: int) -> str:
    skip_tag = str(args.skip_rate).replace(".", "p")
    return (
        f"wikitext2_{model_tag(args.model_path)}_{args.mask_impl_tag}"
        f"_seq{args.seq_len}_pref{args.router_prefix_tokens}"
        f"_m{args.label_samples}_seed{seed}_skip{skip_tag}"
    )


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


def row_from_payload(
    method: str,
    payload: Optional[Dict[str, Any]],
    full_payload: Optional[Dict[str, Any]],
    artifact: Path,
    best_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method,
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
            "eval_tokens": payload.get("eval_tokens"),
            "selected_candidate_distribution": payload.get("selected_candidate_distribution", {}),
        }
    )
    if best_json:
        best = best_json.get("best") or {}
        row.update(
            {
                "best_epoch": best.get("epoch"),
                "validation_nll": best.get("nll"),
                "validation_ppl": best.get("ppl"),
                "validation_unique_masks": best.get("unique_masks"),
            }
        )
    return row


def best_checkpoint_payload(metric_dir: Path, prefix: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Path]:
    best_path = metric_dir / "best_validation_checkpoint.json"
    best_json = load_json(best_path)
    if not best_json:
        return None, None, best_path
    best = best_json.get("best") or {}
    epoch = best.get("epoch")
    if epoch is None:
        return best_json, None, best_path
    test_path = metric_dir / f"{prefix}_best_val_epoch{int(epoch):03d}_test.json"
    return best_json, load_json(test_path), test_path


def collect_seed(args: argparse.Namespace, seed: int) -> Dict[str, Dict[str, Any]]:
    rid = run_id(args, seed)
    root = Path(args.repo_dir)
    result_root = root / "results" / "wikitext2_public_lm_sanity"
    metric_dir = result_root / "metrics" / rid
    related_dir = result_root / "related_metrics" / rid
    opal_val_dir = result_root / "val_ckpt_metrics" / f"{rid}_valckpt"
    raw_val_dir = result_root / "val_ckpt_metrics" / f"{rid}_raw_valckpt"

    full_path = metric_dir / "full_test.json"
    full_payload = load_json(full_path)

    specs: List[Tuple[str, Path, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = [
        ("Full", full_path, full_payload, None),
        ("Static ends_heavy", metric_dir / "static_ends_heavy_test.json", load_json(metric_dir / "static_ends_heavy_test.json"), None),
        ("Static best-on-val", metric_dir / "static_best_on_val_c6_test.json", load_json(metric_dir / "static_best_on_val_c6_test.json"), None),
        ("PuDDing-style", related_dir / "pudding_style_test.json", load_json(related_dir / "pudding_style_test.json"), None),
        ("IG-style", related_dir / "ig_style_test.json", load_json(related_dir / "ig_style_test.json"), None),
        ("layerwise_hidden_router", related_dir / "layerwise_hidden_router_test.json", load_json(related_dir / "layerwise_hidden_router_test.json"), None),
        ("Raw-SetBCE final epoch", metric_dir / "raw_setbce_test.json", load_json(metric_dir / "raw_setbce_test.json"), None),
        ("OPAL-SetBCE final epoch", metric_dir / "opal_setbce_test.json", load_json(metric_dir / "opal_setbce_test.json"), None),
    ]

    raw_best_json, raw_best_payload, raw_best_path = best_checkpoint_payload(raw_val_dir, "raw")
    opal_best_json, opal_best_payload, opal_best_path = best_checkpoint_payload(opal_val_dir, "opal")
    specs.append(("Raw-SetBCE best-on-val", raw_best_path, raw_best_payload, raw_best_json))
    specs.append(("OPAL-SetBCE best-on-val", opal_best_path, opal_best_payload, opal_best_json))

    by_method = {
        method: row_from_payload(method, payload, full_payload, path, best_json=best_json)
        for method, path, payload, best_json in specs
    }
    return {method: by_method.get(method, {"method": method, "status": "missing"}) for method in METHODS}


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


def metric_values(all_rows: Dict[int, Dict[str, Dict[str, Any]]], method: str, key: str) -> List[float]:
    vals = []
    for rows in all_rows.values():
        value = rows.get(method, {}).get(key)
        if finite(value):
            vals.append(float(value))
    return vals


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def wins(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Optional[bool]:
    if not finite(row_a.get("ppl")) or not finite(row_b.get("ppl")):
        return None
    return float(row_a["ppl"]) < float(row_b["ppl"])


def candidate_collapse(row: Dict[str, Any]) -> str:
    dist = row.get("selected_candidate_distribution") or {}
    if not dist:
        return "pending"
    if len(dist) == 1:
        key, value = next(iter(dist.items()))
        return f"{key} ({value})"
    return ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))


def write_seed_table(lines: List[str], seed: int, rows: Dict[str, Dict[str, Any]]) -> None:
    lines.append(f"## Seed {seed}")
    lines.append("")
    lines.append("| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact_skip_count_rate | selected/notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for method in METHODS:
        row = rows[method]
        note = ""
        if method.endswith("best-on-val"):
            note = f"epoch={fmt_int(row.get('best_epoch'))}; val_PPL={fmt(row.get('validation_ppl'))}; val_unique={fmt_int(row.get('validation_unique_masks'))}"
        elif method in {"PuDDing-style", "IG-style"}:
            note = candidate_collapse(row)
        lines.append(
            "| {method} | {nll} | {ppl} | {dnll} | {dppl} | {uniq} | {exact} | {note} |".format(
                method=method,
                nll=fmt(row.get("nll")),
                ppl=fmt(row.get("ppl")),
                dnll=fmt(row.get("delta_nll")),
                dppl=fmt(row.get("delta_ppl")),
                uniq=fmt_int(row.get("unique_masks")),
                exact=fmt(row.get("exact_skip_count_rate"), 3),
                note=note or row.get("status", ""),
            )
        )
    opal_best = rows["OPAL-SetBCE best-on-val"]
    raw_best = rows["Raw-SetBCE best-on-val"]
    layerwise = rows["layerwise_hidden_router"]
    opal_raw_win = wins(opal_best, raw_best)
    opal_layer_win = wins(opal_best, layerwise)
    lines.append("")
    lines.append(f"- OPAL best-on-val wins Raw best-on-val: `{opal_raw_win}`")
    lines.append(f"- OPAL best-on-val wins layerwise_hidden_router: `{opal_layer_win}`")
    lines.append(
        f"- OPAL best-on-val unique_masks: `{fmt_int(opal_best.get('unique_masks'))}`; "
        f"best_epoch: `{fmt_int(opal_best.get('best_epoch'))}`"
    )
    lines.append(f"- PuDDing-style selected candidates: `{candidate_collapse(rows['PuDDing-style'])}`")
    lines.append(f"- IG-style selected candidates: `{candidate_collapse(rows['IG-style'])}`")
    lines.append("")


def write_summary(lines: List[str], all_rows: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    lines.append("## Three-Seed Mean/Std")
    lines.append("")
    lines.append("| method | seeds present | NLL mean | NLL std | PPL mean | PPL std | Delta_NLL mean | Delta_NLL std | Delta_PPL mean | Delta_PPL std | unique_masks mean | unique_masks std | exact-K mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        ppl_vals = metric_values(all_rows, method, "ppl")
        nll_vals = metric_values(all_rows, method, "nll")
        dnll_vals = metric_values(all_rows, method, "delta_nll")
        dppl_vals = metric_values(all_rows, method, "delta_ppl")
        uniq_vals = metric_values(all_rows, method, "unique_masks")
        exact_vals = metric_values(all_rows, method, "exact_skip_count_rate")
        nll_m, nll_s = mean_std(nll_vals)
        ppl_m, ppl_s = mean_std(ppl_vals)
        dnll_m, dnll_s = mean_std(dnll_vals)
        dppl_m, dppl_s = mean_std(dppl_vals)
        uniq_m, uniq_s = mean_std(uniq_vals)
        exact_m, _ = mean_std(exact_vals)
        lines.append(
            f"| {method} | {len(ppl_vals)} | {fmt(nll_m)} | {fmt(nll_s)} | {fmt(ppl_m)} | {fmt(ppl_s)} | "
            f"{fmt(dnll_m)} | {fmt(dnll_s)} | {fmt(dppl_m)} | {fmt(dppl_s)} | {fmt(uniq_m)} | {fmt(uniq_s)} | {fmt(exact_m, 3)} |"
        )
    lines.append("")


def write_judgment(lines: List[str], all_rows: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    opal_raw = []
    opal_layer = []
    opal_uniqs = []
    pudding_collapse = []
    ig_collapse = []
    for seed, rows in all_rows.items():
        win_raw = wins(rows["OPAL-SetBCE best-on-val"], rows["Raw-SetBCE best-on-val"])
        win_layer = wins(rows["OPAL-SetBCE best-on-val"], rows["layerwise_hidden_router"])
        if win_raw is not None:
            opal_raw.append((seed, win_raw))
        if win_layer is not None:
            opal_layer.append((seed, win_layer))
        uniq = rows["OPAL-SetBCE best-on-val"].get("unique_masks")
        if finite(uniq):
            opal_uniqs.append(float(uniq))
        pudding_collapse.append((seed, candidate_collapse(rows["PuDDing-style"])))
        ig_collapse.append((seed, candidate_collapse(rows["IG-style"])))

    opal_mean = mean_std(metric_values(all_rows, "OPAL-SetBCE best-on-val", "ppl"))[0]
    raw_mean = mean_std(metric_values(all_rows, "Raw-SetBCE best-on-val", "ppl"))[0]
    layer_mean = mean_std(metric_values(all_rows, "layerwise_hidden_router", "ppl"))[0]
    opal_mean_wins_raw = opal_mean is not None and raw_mean is not None and opal_mean < raw_mean
    opal_mean_wins_layer = opal_mean is not None and layer_mean is not None and opal_mean < layer_mean

    lines.append("## Required Judgments")
    lines.append("")
    lines.append(f"- OPAL best-on-val wins Raw best-on-val by three-seed mean PPL: `{opal_mean_wins_raw}`")
    lines.append(f"- OPAL best-on-val wins layerwise_hidden_router by three-seed mean PPL: `{opal_mean_wins_layer}`")
    lines.append(f"- Per-seed OPAL vs Raw best-on-val wins: `{opal_raw}`")
    lines.append(f"- Per-seed OPAL vs layerwise wins: `{opal_layer}`")
    lines.append(f"- OPAL best-on-val unique_masks per available seed: `{opal_uniqs}`")
    if opal_uniqs and mean(opal_uniqs) <= 2.0:
        lines.append("- Caveat: OPAL best-on-val unique_masks is very low; describe this as a validation-selected static-like OPAL checkpoint, not a dynamic mask-diversity win.")
    elif opal_uniqs:
        lines.append("- Caveat: inspect OPAL best-on-val unique_masks before claiming dynamic mask diversity.")
    else:
        lines.append("- Caveat: OPAL best-on-val unique_masks pending.")
    lines.append(f"- PuDDing-style selected candidate distributions: `{pudding_collapse}`")
    lines.append(f"- IG-style selected candidate distributions: `{ig_collapse}`")
    lines.append("")
    if opal_mean_wins_raw and opal_mean_wins_layer:
        lines.append("Decision rule: OPAL best-on-val mean currently supports a positive public LM sanity only if the low-unique-mask caveat is acceptable.")
    elif len(opal_raw) == 3 and sum(1 for _, win in opal_raw if win) == 1:
        lines.append("Decision rule: OPAL only wins one seed; keep WikiText-2 in appendix.")
    else:
        lines.append("Decision rule: keep WikiText-2 appendix-level until all three seeds are present and OPAL best-on-val beats Raw/layerwise by mean PPL.")
    lines.append("")


def missing_artifacts(all_rows: Dict[int, Dict[str, Dict[str, Any]]]) -> List[str]:
    missing = []
    for seed, rows in all_rows.items():
        for method, row in rows.items():
            if row.get("status") != "ok":
                missing.append(f"seed={seed} method={method} artifact={row.get('artifact', 'unknown')}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default="/workspace/PriorDynamicPruning")
    parser.add_argument("--model_path", default="/workspace/ckpts/Qwen2.5-1.5B")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 3407])
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--router_prefix_tokens", type=int, default=256)
    parser.add_argument("--label_samples", type=int, default=2000)
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--mask_impl_tag", default="maskcfg")
    parser.add_argument("--output_md", default="/workspace/PriorDynamicPruning/docs/WIKITEXT2_3SEED_PUBLIC_LM_RESULTS.md")
    args = parser.parse_args()

    all_rows = {seed: collect_seed(args, seed) for seed in args.seeds}
    lines: List[str] = [
        "# WikiText-2 3-Seed Public LM Results",
        "",
        "Last updated: generated from server JSON artifacts.",
        "",
        "## Fixed Setup",
        "",
        f"- model: `{args.model_path}`",
        "- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`",
        f"- seeds: `{args.seeds}`",
        f"- seq_len: `{args.seq_len}`",
        f"- router_prefix_tokens: `{args.router_prefix_tokens}`",
        f"- label_samples: `{args.label_samples}`",
        "- eval_windows: `512`",
        f"- skip_rate: `{args.skip_rate}`",
        "- skip_count: `7`",
        "- protected_head / protected_tail: `4 / 2`",
        "",
    ]
    for seed in args.seeds:
        write_seed_table(lines, seed, all_rows[seed])
    write_summary(lines, all_rows)
    write_judgment(lines, all_rows)

    missing = missing_artifacts(all_rows)
    lines.append("## Missing Artifacts")
    lines.append("")
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
