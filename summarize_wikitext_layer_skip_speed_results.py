#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", action="append", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    payloads = [load_json(path) for path in args.input_json]
    lines: List[str] = [
        "# WikiText-2 Layer-Skip Speed Results",
        "",
        "This benchmark measures real compute skipping under three batching regimes:",
        "",
        "- `batch1_true_skip`: one request at a time; skipped layers return before attention/MLP compute.",
        "- `grouped_by_mask`: requests are grouped by identical keep mask, so each group can skip whole layers.",
        "- `mixed_batch_naive`: ordinary batching with per-sample masks; a layer is physically skipped only when every row in the batch skips it.",
        "",
        "The paper speed table should use `batch1_true_skip` and `grouped_by_mask`. "
        "`mixed_batch_naive` is a diagnostic showing why ungrouped dynamic masks do not realize the full theoretical speedup.",
        "",
    ]

    for payload in payloads:
        rows = payload["rows"]
        lines.extend(
            [
                f"## {payload['model_label']}",
                "",
                f"- model path: `{payload['model']}`",
                f"- run label: `{payload['run_label']}`",
                f"- seq_len / router_prefix_tokens: `{payload['seq_len']}` / `{payload['router_prefix_tokens']}`",
                f"- windows / batch size: `{payload['num_windows']}` / `{payload['batch_size']}`",
                f"- skip_count / keep_count: `{payload['skip_count']}` / `{payload['keep_count']}`",
                "",
                "| mode | method | PPL ref | windows/s | tok/s | latency/window ms | router ms/win | forward ms/win | router % | speedup | avg kept | unique masks | exact-K |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        mode_order = {"batch1_true_skip": 0, "grouped_by_mask": 1, "mixed_batch_naive": 2}
        for row in sorted(rows, key=lambda x: (mode_order.get(x["mode"], 99), x["method_slug"])):
            lines.append(
                "| {mode} | {method} | {ppl} | {wps} | {tps} | {lat} | {router} | {forward} | {router_pct}% | {speedup} | {kept} | {uniq} | {exact} |".format(
                    mode=row["mode"],
                    method=row["method_label"],
                    ppl=fmt(row.get("reference_ppl"), 4),
                    wps=fmt(row.get("windows_per_sec"), 4),
                    tps=fmt(row.get("input_tokens_per_sec"), 1),
                    lat=fmt(1000.0 * float(row.get("latency_sec_per_window", 0.0)), 2),
                    router=fmt(row.get("mask_inference_ms_per_window"), 2),
                    forward=fmt(row.get("model_forward_ms_per_window"), 2),
                    router_pct=fmt(100.0 * float(row.get("mask_inference_pct", 0.0)), 1),
                    speedup="NA"
                    if row.get("speedup_vs_full_same_mode") is None
                    else f"{float(row['speedup_vs_full_same_mode']):.3f}x",
                    kept=fmt(row.get("average_kept_layers"), 2),
                    uniq=int(row.get("unique_masks", 0)),
                    exact=fmt(row.get("exact_skip_count_rate"), 3),
                )
            )
        lines.append("")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
