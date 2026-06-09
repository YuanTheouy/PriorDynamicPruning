#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)
sys.path.insert(0, current_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_wikitext_opal_ppl import load_mask_runtime, keep_masks_for_runtime
from wikitext_opal_utils import (
    allowed_layers_from_policy,
    clear_custom_policy,
    collate_wikitext_windows,
    detect_num_layers,
    dtype_from_precision,
    load_wikitext_token_ids,
    mask_key,
    resolve_skip_budget,
    set_custom_policy,
    static_keep_mask,
    summarize_keep_masks,
    WikitextWindowDataset,
)


def load_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    cuda_sync()
    start = time.perf_counter()
    result = fn()
    cuda_sync()
    return float(time.perf_counter() - start), result


def chunked(values: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def sanitize_spec(spec: Dict[str, object]) -> Dict[str, object]:
    spec = dict(spec)
    metric_path = str(spec.get("metric_json") or "")
    metric = load_json(metric_path) if metric_path else {}
    if metric:
        spec.setdefault("method", metric.get("method"))
        spec.setdefault("label", metric.get("method_label"))
        spec.setdefault("static_strategy", metric.get("static_strategy"))
        spec.setdefault("risk_router_ckpt", metric.get("risk_router_ckpt"))
        spec.setdefault("candidate_router_ckpt", metric.get("candidate_router_ckpt"))
        spec.setdefault("ig_artifact", metric.get("ig_artifact"))
        spec.setdefault("ppl", metric.get("ppl"))
        spec.setdefault("nll", metric.get("nll"))
        spec.setdefault("average_kept_layers", metric.get("average_kept_layers"))
        spec.setdefault("unique_masks_reference", metric.get("unique_masks"))
    method = str(spec.get("method") or "")
    if method == "full":
        spec["label"] = spec.get("label") or "Full"
    if method == "static":
        spec["static_strategy"] = spec.get("static_strategy") or "ends_heavy"
        spec["label"] = spec.get("label") or f"Static {spec['static_strategy']}"
    if method == "router":
        if not spec.get("risk_router_ckpt"):
            raise ValueError(f"Router spec lacks risk_router_ckpt or metric_json with risk_router_ckpt: {spec}")
        spec["label"] = spec.get("label") or "Router"
    if method == "candidate_router":
        if not spec.get("candidate_router_ckpt"):
            raise ValueError(
                f"Candidate-router spec lacks candidate_router_ckpt or metric_json with it: {spec}"
            )
        spec["label"] = spec.get("label") or "PuDDing-style"
    if method == "ig":
        if not spec.get("ig_artifact"):
            raise ValueError(f"IG spec lacks ig_artifact or metric_json with it: {spec}")
        spec["label"] = spec.get("label") or "IG-style"
    if "slug" not in spec:
        spec["slug"] = str(spec["label"]).lower().replace(" ", "_").replace("/", "_")
    return spec


def build_batches(dataset, router_prefix_tokens: int, batch_size: int, device) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for rows in chunked([dataset[idx] for idx in range(len(dataset))], batch_size):
        batch = collate_wikitext_windows(rows, router_prefix_tokens)
        batches.append(
            {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "sample_id": batch["sample_id"],
                "window_start": batch["window_start"],
            }
        )
    return batches


def forward_once(model, input_ids, attention_mask, mask_payload=None) -> None:
    if mask_payload is None:
        clear_custom_policy(model)
    else:
        set_custom_policy(model, mask_payload)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            _ = outputs.logits[:, -1, :1].float().sum()
            del outputs
    finally:
        clear_custom_policy(model)


def masks_for_batch(runtime, model, batch, router_prefix_tokens: int, num_layers: int) -> List[List[int]]:
    keep_masks, _ = keep_masks_for_runtime(
        runtime,
        model,
        batch["input_ids"],
        batch["attention_mask"],
        router_prefix_tokens,
        num_layers,
    )
    return [[int(v) for v in mask] for mask in keep_masks]


def attach_timing_breakdown(
    row: Dict[str, object],
    mask_inference_durations: Sequence[float],
    forward_durations: Sequence[float],
) -> Dict[str, object]:
    mask_sec = float(sum(mask_inference_durations))
    forward_sec = float(sum(forward_durations))
    total_sec = float(row.get("total_sec", mask_sec + forward_sec))
    num_windows = max(1, int(row.get("num_windows", 0)))
    row["mask_inference_sec"] = mask_sec
    row["model_forward_sec"] = forward_sec
    row["mask_inference_ms_per_window"] = float(1000.0 * mask_sec / num_windows)
    row["model_forward_ms_per_window"] = float(1000.0 * forward_sec / num_windows)
    row["mask_inference_pct"] = float(mask_sec / total_sec) if total_sec > 0 else 0.0
    row["model_forward_pct"] = float(forward_sec / total_sec) if total_sec > 0 else 0.0
    return row


def run_batch1_mode(args, model, spec, runtime, batches, num_layers: int) -> Dict[str, object]:
    sample_batches = []
    for batch in batches:
        for row_idx in range(batch["input_ids"].size(0)):
            sample_batches.append(
                {
                    "input_ids": batch["input_ids"][row_idx : row_idx + 1],
                    "attention_mask": batch["attention_mask"][row_idx : row_idx + 1],
                }
            )
    sample_batches = sample_batches[: int(args.batch1_windows or args.num_windows)]
    durations: List[float] = []
    mask_inference_durations: List[float] = []
    forward_durations: List[float] = []
    masks: List[List[int]] = []

    def run_one(sample_batch, record: bool) -> None:
        if spec["method"] == "full":
            elapsed, _ = timed(lambda: forward_once(model, sample_batch["input_ids"], sample_batch["attention_mask"]))
            if record:
                durations.append(elapsed)
                mask_inference_durations.append(0.0)
                forward_durations.append(elapsed)
                masks.append([1] * num_layers)
            return

        mask_elapsed, keep_masks = timed(
            lambda: masks_for_batch(runtime, model, sample_batch, args.router_prefix_tokens, num_layers)
        )
        forward_elapsed, _ = timed(
            lambda: forward_once(model, sample_batch["input_ids"], sample_batch["attention_mask"], keep_masks[0])
        )
        keep_mask = keep_masks[0]
        if record:
            durations.append(mask_elapsed + forward_elapsed)
            mask_inference_durations.append(mask_elapsed)
            forward_durations.append(forward_elapsed)
            masks.append(keep_mask)

    for sample_batch in sample_batches[: int(args.warmup_windows)]:
        run_one(sample_batch, record=False)
    for sample_batch in sample_batches:
        run_one(sample_batch, record=True)

    row = summarize_speed_row(args, spec, "batch1_true_skip", durations, masks, num_layers)
    return attach_timing_breakdown(row, mask_inference_durations, forward_durations)


def run_mixed_batch_mode(args, model, spec, runtime, batches, num_layers: int) -> Dict[str, object]:
    durations: List[float] = []
    mask_inference_durations: List[float] = []
    forward_durations: List[float] = []
    masks: List[List[int]] = []

    def run_one(batch, record: bool) -> None:
        if spec["method"] == "full":
            elapsed, _ = timed(lambda: forward_once(model, batch["input_ids"], batch["attention_mask"]))
            keep_masks = [[1] * num_layers for _ in range(batch["input_ids"].size(0))]
            mask_elapsed = 0.0
            forward_elapsed = elapsed
        else:
            mask_elapsed, keep_masks = timed(
                lambda: masks_for_batch(runtime, model, batch, args.router_prefix_tokens, num_layers)
            )
            mask_payload = keep_masks
            if all(mask == keep_masks[0] for mask in keep_masks):
                mask_payload = keep_masks[0]
            forward_elapsed, _ = timed(
                lambda: forward_once(model, batch["input_ids"], batch["attention_mask"], mask_payload)
            )
            elapsed = mask_elapsed + forward_elapsed
        if record:
            durations.append(elapsed)
            mask_inference_durations.append(mask_elapsed)
            forward_durations.append(forward_elapsed)
            masks.extend(keep_masks)

    for batch in batches[: int(args.warmup_batches)]:
        run_one(batch, record=False)
    for batch in batches:
        run_one(batch, record=True)

    row = summarize_speed_row(args, spec, "mixed_batch_naive", durations, masks, num_layers)
    return attach_timing_breakdown(row, mask_inference_durations, forward_durations)


def run_grouped_mode(args, model, spec, runtime, batches, num_layers: int) -> Dict[str, object]:
    if spec["method"] == "full":
        return run_mixed_batch_mode(args, model, spec, runtime, batches, num_layers) | {
            "mode": "grouped_by_mask",
            "mask_inference_sec": 0.0,
        }

    all_records = []
    mask_inference_durations: List[float] = []
    all_masks: List[List[int]] = []

    for batch in batches[: int(args.warmup_batches)]:
        _ = masks_for_batch(runtime, model, batch, args.router_prefix_tokens, num_layers)

    for batch in batches:
        elapsed, keep_masks = timed(
            lambda batch=batch: masks_for_batch(runtime, model, batch, args.router_prefix_tokens, num_layers)
        )
        mask_inference_durations.append(elapsed)
        all_masks.extend(keep_masks)
        for row_idx, keep_mask in enumerate(keep_masks):
            all_records.append(
                {
                    "key": mask_key(keep_mask),
                    "mask": keep_mask,
                    "input_ids": batch["input_ids"][row_idx : row_idx + 1],
                    "attention_mask": batch["attention_mask"][row_idx : row_idx + 1],
                }
            )

    groups = defaultdict(list)
    for record in all_records:
        groups[record["key"]].append(record)

    forward_durations: List[float] = []
    group_batches = []
    for records in groups.values():
        keep_mask = records[0]["mask"]
        for sub_records in chunked(records, args.batch_size):
            input_ids = torch.cat([row["input_ids"] for row in sub_records], dim=0)
            attention_mask = torch.cat([row["attention_mask"] for row in sub_records], dim=0)
            group_batches.append((keep_mask, input_ids, attention_mask))

    for keep_mask, input_ids, attention_mask in group_batches[: int(args.warmup_batches)]:
        forward_once(model, input_ids, attention_mask, keep_mask)
    for keep_mask, input_ids, attention_mask in group_batches:
        elapsed, _ = timed(lambda km=keep_mask, ids=input_ids, am=attention_mask: forward_once(model, ids, am, km))
        forward_durations.append(elapsed)

    total_durations = mask_inference_durations + forward_durations
    row = summarize_speed_row(args, spec, "grouped_by_mask", total_durations, all_masks, num_layers)
    attach_timing_breakdown(row, mask_inference_durations, forward_durations)
    row["grouped_forward_sec"] = float(sum(forward_durations))
    row["group_count"] = int(len(groups))
    row["group_forward_batches"] = int(len(group_batches))
    return row


def summarize_speed_row(
    args,
    spec: Dict[str, object],
    mode: str,
    durations: Sequence[float],
    masks: Sequence[Sequence[int]],
    num_layers: int,
) -> Dict[str, object]:
    total_sec = float(sum(durations))
    num_windows = int(len(masks))
    input_tokens = int(num_windows * int(args.seq_len))
    mask_summary = summarize_keep_masks(
        masks,
        expected_skip_count=0 if spec["method"] == "full" else int(args.skip_count),
    )
    return {
        "model": args.model,
        "model_label": args.model_label,
        "run_label": args.run_label,
        "method_slug": spec["slug"],
        "method_label": spec["label"],
        "method": spec["method"],
        "mode": mode,
        "num_windows": num_windows,
        "seq_len": int(args.seq_len),
        "router_prefix_tokens": int(args.router_prefix_tokens),
        "batch_size": 1 if mode == "batch1_true_skip" else int(args.batch_size),
        "total_sec": total_sec,
        "mean_batch_sec": float(statistics.mean(durations)) if durations else 0.0,
        "p50_batch_sec": float(statistics.median(durations)) if durations else 0.0,
        "windows_per_sec": float(num_windows / total_sec) if total_sec > 0 else 0.0,
        "input_tokens_per_sec": float(input_tokens / total_sec) if total_sec > 0 else 0.0,
        "latency_sec_per_window": float(total_sec / max(1, num_windows)),
        "input_tokens": input_tokens,
        "unique_masks": int(mask_summary["unique_masks"]),
        "average_kept_layers": float(mask_summary["average_kept_layers"]),
        "average_skipped_layers": float(mask_summary["average_skipped_layers"]),
        "exact_skip_count_rate": float(mask_summary["exact_skip_count_rate"]),
        "mask_usage": mask_summary["mask_usage"],
        "reference_ppl": spec.get("ppl"),
        "reference_nll": spec.get("nll"),
        "risk_router_ckpt": spec.get("risk_router_ckpt", ""),
        "candidate_router_ckpt": spec.get("candidate_router_ckpt", ""),
        "ig_artifact": spec.get("ig_artifact", ""),
        "static_strategy": spec.get("static_strategy"),
    }


def add_speedups(rows: List[Dict[str, object]]) -> None:
    full_by_mode = {}
    for row in rows:
        if row["method"] == "full":
            full_by_mode[row["mode"]] = float(row["total_sec"])
    for row in rows:
        base = full_by_mode.get(row["mode"], 0.0)
        row["speedup_vs_full_same_mode"] = float(base / float(row["total_sec"])) if base > 0 else None


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    rows = payload["rows"]
    lines = [
        "# WikiText-2 Layer-Skip Speed Benchmark",
        "",
        f"- model: `{payload['model']}`",
        f"- run label: `{payload['run_label']}`",
        f"- seq_len / router_prefix_tokens: `{payload['seq_len']}` / `{payload['router_prefix_tokens']}`",
        f"- measured windows: `{payload['num_windows']}`",
        f"- batch size: `{payload['batch_size']}`",
        f"- modes: `batch1_true_skip`, `mixed_batch_naive`, `grouped_by_mask`",
        "",
        "Important: `batch1_true_skip` and `grouped_by_mask` measure real skipped compute. "
        "`mixed_batch_naive` is diagnostic because per-sample-different masks only skip a layer when all rows in the batch skip it.",
        "",
        "| mode | method | PPL ref | windows/s | tok/s | latency/window ms | router ms/win | forward ms/win | router % | speedup | avg kept | unique masks | exact-K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"batch1_true_skip": 0, "grouped_by_mask": 1, "mixed_batch_naive": 2}
    for row in sorted(rows, key=lambda r: (order.get(r["mode"], 99), r["method_slug"])):
        ppl = row.get("reference_ppl")
        speedup = row.get("speedup_vs_full_same_mode")
        lines.append(
            "| {mode} | {method} | {ppl} | {wps:.4f} | {tps:.1f} | {lat:.2f} | {router:.2f} | {forward:.2f} | {router_pct:.1f}% | {speedup} | {kept:.2f} | {uniq} | {exact:.3f} |".format(
                mode=row["mode"],
                method=row["method_label"],
                ppl="NA" if ppl is None else f"{float(ppl):.4f}",
                wps=float(row["windows_per_sec"]),
                tps=float(row["input_tokens_per_sec"]),
                lat=1000.0 * float(row["latency_sec_per_window"]),
                router=float(row.get("mask_inference_ms_per_window", 0.0)),
                forward=float(row.get("model_forward_ms_per_window", 0.0)),
                router_pct=100.0 * float(row.get("mask_inference_pct", 0.0)),
                speedup="NA" if speedup is None else f"{float(speedup):.3f}x",
                kept=float(row["average_kept_layers"]),
                uniq=int(row["unique_masks"]),
                exact=float(row["exact_skip_count_rate"]),
            )
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_label", default="")
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--dataset_disk_path", default="")
    parser.add_argument("--dataset_cache_dir", default="")
    parser.add_argument("--dataset_path", default="wikitext")
    parser.add_argument("--dataset_name", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq_len", type=int, default=1536)
    parser.add_argument("--router_prefix_tokens", type=int, default=1024)
    parser.add_argument("--num_windows", type=int, default=96)
    parser.add_argument("--batch1_windows", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_windows", type=int, default=2)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--skip_count", type=int, default=0)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods_json", default="")
    parser.add_argument("--methods_json_file", default="")
    parser.add_argument("--modes", default="batch1_true_skip,grouped_by_mask,mixed_batch_naive")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    args.model_label = args.model_label or os.path.basename(os.path.normpath(args.model))
    if args.methods_json_file:
        method_specs = json.loads(Path(args.methods_json_file).read_text(encoding="utf-8"))
    elif args.methods_json:
        method_specs = json.loads(args.methods_json)
    else:
        raise ValueError("--methods_json or --methods_json_file is required")
    method_specs = [sanitize_spec(spec) for spec in method_specs]
    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    token_ids = load_wikitext_token_ids(
        tokenizer,
        split=args.split,
        dataset_disk_path=args.dataset_disk_path,
        dataset_cache_dir=args.dataset_cache_dir,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
    )
    dataset = WikitextWindowDataset(
        token_ids,
        seq_len=args.seq_len,
        max_windows=args.num_windows,
        sample_strategy="first",
        sample_seed=args.seed,
    )
    batches = build_batches(dataset, args.router_prefix_tokens, args.batch_size, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    resolved_skip_count, keep_count = resolve_skip_budget(num_layers, args.skip_rate, args.skip_count)
    args.skip_count = int(resolved_skip_count)
    allowed_layers = allowed_layers_from_policy(num_layers, args.protected_head, args.protected_tail)
    hidden_size = int(model.config.hidden_size)

    rows: List[Dict[str, object]] = []
    for spec in method_specs:
        method = str(spec["method"])
        runtime = None
        if method != "full":
            runtime = load_mask_runtime(
                spec,
                model,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
                default_skip_count=resolved_skip_count,
                default_allowed_layers=allowed_layers,
                protected_head=args.protected_head,
                protected_tail=args.protected_tail,
                seed=args.seed,
            )
        print(f"=== Benchmark {spec['label']} ===", flush=True)
        if "batch1_true_skip" in modes:
            rows.append(run_batch1_mode(args, model, spec, runtime, batches, num_layers))
        if "grouped_by_mask" in modes:
            rows.append(run_grouped_mode(args, model, spec, runtime, batches, num_layers))
        if "mixed_batch_naive" in modes:
            rows.append(run_mixed_batch_mode(args, model, spec, runtime, batches, num_layers))

    add_speedups(rows)
    payload = {
        "model": args.model,
        "model_label": args.model_label,
        "run_label": args.run_label,
        "split": args.split,
        "seq_len": int(args.seq_len),
        "router_prefix_tokens": int(args.router_prefix_tokens),
        "num_windows": int(len(dataset)),
        "batch_size": int(args.batch_size),
        "batch1_windows": int(args.batch1_windows),
        "precision": args.precision,
        "num_layers": int(num_layers),
        "skip_rate": float(args.skip_rate),
        "skip_count": int(resolved_skip_count),
        "keep_count": int(keep_count),
        "protected_head": int(args.protected_head),
        "protected_tail": int(args.protected_tail),
        "method_specs": method_specs,
        "rows": rows,
    }
    write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
