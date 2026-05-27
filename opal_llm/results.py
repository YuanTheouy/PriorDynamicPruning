import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .mask_library import summarize_masks
from .quality_metrics import summarize_quality_metrics


DEFAULT_TOPK = (1, 3, 5, 10, 20, 50)


def result_dirs(output_dir: str) -> Dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "raw_json": root / "raw_json",
        "logs": root / "logs",
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def command_line() -> str:
    return " ".join([sys.executable] + sys.argv)


def git_commit(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _dedup_value(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def load_ground_truths(test_file: str, drop_dedup: bool = True) -> List[str]:
    with open(test_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if drop_dedup and rows and "dedup" in rows[0]:
            rows = [row for row in rows if not _dedup_value(row.get("dedup"))]
        if not rows:
            return []
        field = "item_sid" if "item_sid" in rows[0] else reader.fieldnames[-1]
        return [str(row[field]).strip(' \n"') for row in rows]


def load_valid_sids(info_file: str) -> List[str]:
    valid = []
    with open(info_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            valid.append(line.split("\t")[0].strip())
    return valid


def rank_of_target(predictions: Sequence[str], target: str) -> int:
    target = str(target).strip(' \n"')
    for idx, pred in enumerate(predictions):
        if str(pred).strip(' \n"') == target:
            return idx
    return 1_000_000


def compute_metrics(
    predictions: Sequence[Dict[str, object]],
    ground_truths: Sequence[str],
    valid_sids: Optional[Iterable[str]] = None,
    topk: Sequence[int] = DEFAULT_TOPK,
) -> Dict[str, float]:
    if not predictions or not ground_truths:
        return {f"NDCG@{k}": 0.0 for k in topk} | {f"HR@{k}": 0.0 for k in topk}

    valid_set = set(valid_sids or [])
    indexed_rows = []
    for row in sorted(predictions, key=lambda item: int(item.get("index", 0))):
        sample_index = int(row.get("index", -1))
        if 0 <= sample_index < len(ground_truths):
            indexed_rows.append((row, ground_truths[sample_index]))

    ndcg = {k: 0.0 for k in topk}
    hr = {k: 0.0 for k in topk}
    total_preds = 0
    invalid_preds = 0
    failed_samples = 0

    for row, target in indexed_rows:
        sample_predictions = [str(x).strip(' \n"') for x in row.get("sample_predictions", [])]
        if not sample_predictions:
            failed_samples += 1
        if valid_set:
            total_preds += len(sample_predictions)
            invalid_preds += sum(1 for pred in sample_predictions if pred not in valid_set)

        rank = rank_of_target(sample_predictions, target)
        for k in topk:
            if rank < k:
                ndcg[k] += 1.0 / math.log(rank + 2)
                hr[k] += 1.0

    denom = max(1, len(indexed_rows))
    metrics = {}
    for k in topk:
        metrics[f"NDCG@{k}"] = ndcg[k] / denom / (1.0 / math.log(2))
        metrics[f"HR@{k}"] = hr[k] / denom
    metrics["invalid_prediction_rate"] = invalid_preds / total_preds if total_preds else 0.0
    metrics["constrained_failure_rate"] = failed_samples / denom
    metrics["num_samples"] = len(indexed_rows)
    return metrics


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_summary_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_payload(
    metadata: Dict[str, object],
    predictions: Sequence[Dict[str, object]],
    ground_truths: Sequence[str],
    valid_sids: Sequence[str],
    latency: Dict[str, object],
    timing_records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    masks = [row.get("layer_mask", []) for row in predictions if row.get("layer_mask") is not None]
    mask_ids = [
        str(row.get("mask_id") or row.get("template_id"))
        for row in predictions
        if row.get("mask_id") is not None or row.get("template_id") is not None
    ]
    metrics = compute_metrics(predictions, ground_truths, valid_sids)
    full_predictions = []
    for row in predictions:
        if row.get("full_sample_predictions"):
            full_row = dict(row)
            full_row["sample_predictions"] = row.get("full_sample_predictions", [])
            full_predictions.append(full_row)
    full_metrics = compute_metrics(full_predictions, ground_truths, valid_sids) if full_predictions else {}
    retention_metrics = {}
    for key, value in metrics.items():
        if key.startswith(("NDCG@", "HR@")):
            full_value = float(full_metrics.get(key, value if metadata.get("method") == "full" else 0.0) or 0.0)
            retention_metrics[f"retention_{key}"] = float(value) / full_value if full_value > 0 else 0.0
            retention_metrics[f"{key}_full"] = full_value
            retention_metrics[f"{key}_skip"] = float(value or 0.0)
    metrics.update(retention_metrics)
    quality_rows = [row for row in predictions if row.get("NLL_full") is not None]
    quality = summarize_quality_metrics(quality_rows)
    for key, value in quality.items():
        metrics[key] = value
    for source_key, metric_key in [
        ("prefix_to_oracle_agreement", "prefix_to_oracle_agreement"),
        ("hamming_distance", "hamming_distance"),
        ("mask_regret", "oracle_regret"),
    ]:
        values = [
            float(row[source_key])
            for row in predictions
            if row.get(source_key) is not None and math.isfinite(float(row[source_key]))
        ]
        if values:
            metrics[metric_key] = sum(values) / len(values)
    serving_records = [
        row.get("metadata", {})
        for row in timing_records
        if isinstance(row.get("metadata"), dict) and row.get("metadata")
    ]
    batch_records = [row for row in serving_records if row.get("batch_size") is not None]
    return {
        "metadata": metadata,
        "metrics": metrics,
        "full_metrics": full_metrics,
        "quality": quality,
        "latency": latency,
        "mask_stats": summarize_masks(masks, mask_ids if mask_ids else None),
        "serving_stats": summarize_serving_records(batch_records),
        "timing_records": list(timing_records),
        "predictions": list(predictions),
    }


def summarize_serving_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {
            "mean_unique_masks_in_batch": 0.0,
            "mean_mask_entropy_in_batch": 0.0,
            "grouping_strategy_usage": {},
            "mean_num_groups": 0.0,
        }
    strategies = [str(row.get("grouping_strategy", "none")) for row in records]
    return {
        "mean_unique_masks_in_batch": sum(float(row.get("unique_masks_in_batch", 0.0)) for row in records)
        / len(records),
        "mean_mask_entropy_in_batch": sum(float(row.get("mask_entropy_in_batch", 0.0)) for row in records)
        / len(records),
        "grouping_strategy_usage": {key: strategies.count(key) for key in sorted(set(strategies))},
        "mean_num_groups": sum(float(row.get("num_groups", 0.0)) for row in records) / len(records),
    }


def summary_row(payload: Dict[str, object], raw_output_path: str) -> Dict[str, object]:
    metadata = payload.get("metadata", {})
    metrics = payload.get("metrics", {})
    latency = payload.get("latency", {})
    mask_stats = payload.get("mask_stats", {})
    serving_stats = payload.get("serving_stats", {})
    return {
        "dataset": metadata.get("dataset"),
        "method": metadata.get("method"),
        "opal_stage": metadata.get("opal_stage"),
        "skip_rate": metadata.get("skip_rate"),
        "budget": metadata.get("top_k_layers"),
        "checkpoint": metadata.get("checkpoint"),
        "seed": metadata.get("seed"),
        "git_commit": metadata.get("git_commit"),
        "NDCG@10": metrics.get("NDCG@10"),
        "NDCG@50": metrics.get("NDCG@50"),
        "HR@10": metrics.get("HR@10"),
        "HR@50": metrics.get("HR@50"),
        "retention_NDCG@10": metrics.get("retention_NDCG@10"),
        "retention_NDCG@50": metrics.get("retention_NDCG@50"),
        "retention_HR@10": metrics.get("retention_HR@10"),
        "retention_HR@50": metrics.get("retention_HR@50"),
        "NLL_full": metrics.get("NLL_full"),
        "NLL_skip": metrics.get("NLL_skip"),
        "Delta_NLL": metrics.get("Delta_NLL"),
        "PPL_full": metrics.get("PPL_full"),
        "PPL_skip": metrics.get("PPL_skip"),
        "Delta_PPL": metrics.get("Delta_PPL"),
        "KL_full_to_skip": metrics.get("KL_full_to_skip"),
        "oracle_regret": metrics.get("oracle_regret"),
        "prefix_to_oracle_agreement": metrics.get("prefix_to_oracle_agreement"),
        "hamming_distance": metrics.get("hamming_distance"),
        "invalid_prediction_rate": metrics.get("invalid_prediction_rate"),
        "latency_mean_sec": latency.get("latency_mean_sec"),
        "latency_std_sec": latency.get("latency_std_sec"),
        "latency_p50_sec": latency.get("latency_p50_sec"),
        "latency_p95_sec": latency.get("latency_p95_sec"),
        "throughput_samples_per_sec": latency.get("throughput_samples_per_sec"),
        "speedup_vs_full": metadata.get("speedup_vs_full"),
        "average_kept_layers": mask_stats.get("average_kept_layers"),
        "unique_masks": mask_stats.get("unique_masks"),
        "mask_entropy": mask_stats.get("mask_entropy"),
        "per_batch_unique_masks": serving_stats.get("mean_unique_masks_in_batch"),
        "per_batch_mask_entropy": serving_stats.get("mean_mask_entropy_in_batch"),
        "router_latency_sec": latency.get("router_latency_sec"),
        "compensation_latency_sec": latency.get("compensation_latency_sec"),
        "raw_output_path": raw_output_path,
    }
