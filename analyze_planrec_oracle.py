#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence

from opal_llm.results import load_ground_truths, rank_of_target, result_dirs, write_json
from opal_llm.template_library import assignment_entropy


MISSING_RANK = 1_000_000


def load_payload(path: str) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "predictions" not in payload:
        raise ValueError(f"{path} is not a final OPAL result JSON")
    return payload


def prediction_map(payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(row.get("index", 0)): row for row in payload.get("predictions", [])}


def oracle_candidate_rank(row: Dict[str, object], template_id: str):
    for candidate in row.get("oracle_candidates", []) or []:
        if str(candidate.get("template_id")) == str(template_id):
            return int(candidate.get("rank", MISSING_RANK))
    return None


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(len(ordered) - 1, lower + 1)
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def analyze(oracle_payload, prediction_payload, ground_truths: Sequence[str]) -> Dict[str, object]:
    oracle_rows = prediction_map(oracle_payload)
    prediction_rows = prediction_map(prediction_payload)
    common_indices = sorted(set(oracle_rows) & set(prediction_rows))
    oracle_template_ids = [str(oracle_rows[idx].get("template_id")) for idx in common_indices]
    pred_template_ids = [str(prediction_rows[idx].get("template_id")) for idx in common_indices]
    oracle_counts = Counter(oracle_template_ids)

    agreements = []
    rank_regrets = []
    selected_ranks = []
    oracle_ranks = []
    fallback_dynamic_rank_count = 0
    missing_candidate_count = 0

    for idx in common_indices:
        oracle_row = oracle_rows[idx]
        pred_row = prediction_rows[idx]
        oracle_template = str(oracle_row.get("template_id"))
        pred_template = str(pred_row.get("template_id"))
        agreements.append(1.0 if oracle_template == pred_template else 0.0)

        oracle_rank_value = oracle_row.get("oracle_rank")
        if oracle_rank_value is None:
            oracle_rank_value = rank_of_target(oracle_row.get("sample_predictions", []), "")
        oracle_rank = int(oracle_rank_value)
        selected_rank = oracle_candidate_rank(oracle_row, pred_template)
        if selected_rank is None:
            missing_candidate_count += 1
            if idx < len(ground_truths):
                selected_rank = rank_of_target(pred_row.get("sample_predictions", []), ground_truths[idx])
                fallback_dynamic_rank_count += 1
            else:
                selected_rank = MISSING_RANK
        oracle_ranks.append(oracle_rank)
        selected_ranks.append(selected_rank)
        rank_regrets.append(float(selected_rank) - float(oracle_rank))

    top1_share = 0.0
    if common_indices and oracle_counts:
        top1_share = max(oracle_counts.values()) / float(len(common_indices))

    return {
        "num_common_samples": len(common_indices),
        "oracle_unique_templates": len(oracle_counts),
        "oracle_template_usage": dict(oracle_counts),
        "oracle_template_entropy": assignment_entropy(oracle_template_ids),
        "oracle_top1_template_share": top1_share,
        "prediction_unique_templates": len(set(pred_template_ids)),
        "prediction_template_entropy": assignment_entropy(pred_template_ids),
        "prefix_to_oracle_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
        "mean_oracle_rank": sum(oracle_ranks) / len(oracle_ranks) if oracle_ranks else 0.0,
        "mean_selected_rank": sum(selected_ranks) / len(selected_ranks) if selected_ranks else 0.0,
        "mean_template_rank_regret": sum(rank_regrets) / len(rank_regrets) if rank_regrets else 0.0,
        "median_template_rank_regret": float(median(rank_regrets)) if rank_regrets else 0.0,
        "p95_template_rank_regret": percentile(rank_regrets, 0.95),
        "missing_candidate_count": missing_candidate_count,
        "fallback_dynamic_rank_count": fallback_dynamic_rank_count,
        "regret_source": "oracle_candidates_if_available_else_prediction_rank",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute OPAL oracle diversity, prefix-to-oracle agreement, and template regret."
    )
    parser.add_argument("--oracle_json", required=True)
    parser.add_argument("--prediction_json", required=True, help="Dynamic, input-guided, or layerwise result JSON.")
    parser.add_argument("--test_file", default="", help="Needed only when oracle candidates were not saved.")
    parser.add_argument("--output_dir", default="./results/planrec_experiments")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    oracle_payload = load_payload(args.oracle_json)
    prediction_payload = load_payload(args.prediction_json)
    ground_truths: List[str] = load_ground_truths(args.test_file, drop_dedup=False) if args.test_file else []
    analysis = analyze(oracle_payload, prediction_payload, ground_truths)

    oracle_meta = oracle_payload.get("metadata", {})
    pred_meta = prediction_payload.get("metadata", {})
    payload = {
        "metadata": {
            "oracle_json": args.oracle_json,
            "prediction_json": args.prediction_json,
            "oracle_method": oracle_meta.get("method"),
            "prediction_method": pred_meta.get("method"),
            "dataset": pred_meta.get("dataset") or oracle_meta.get("dataset"),
            "budget": pred_meta.get("top_k_layers") or oracle_meta.get("top_k_layers"),
            "prediction_run": Path(args.prediction_json).stem,
            "oracle_run": Path(args.oracle_json).stem,
        },
        "analysis": analysis,
    }

    dirs = result_dirs(args.output_dir)
    name = args.name or f"oracle_analysis_{payload['metadata']['dataset']}_{payload['metadata']['prediction_method']}_k{payload['metadata']['budget']}"
    json_path = dirs["raw_json"] / f"{name}.json"
    md_path = dirs["tables"] / f"{name}.md"
    write_json(json_path, payload)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| metric | value |\n")
        f.write("|---|---:|\n")
        for key, value in analysis.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, float) and (math.isfinite(value)):
                f.write(f"| {key} | {value:.6f} |\n")
            else:
                f.write(f"| {key} | {value} |\n")
    print(f"Wrote oracle analysis JSON to {json_path}")
    print(f"Wrote oracle analysis table to {md_path}")


if __name__ == "__main__":
    main()
