#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


OBJECTIVE_TO_QUALITY_KEY = {
    "Delta_NLL": "Delta_NLL",
    "KL": "KL_full_to_skip",
}


def metadata_path_for(candidate_label_file: str, candidate_metadata_file: str = "") -> Path:
    if candidate_metadata_file:
        return Path(candidate_metadata_file)
    return Path(candidate_label_file + ".metadata.json")


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No candidate-label rows found in {path}")
    return rows


def loss_row(row, objective: str):
    if row.get("objective") == objective or not row.get("candidate_quality"):
        return [float(v) for v in row["candidate_losses"]]
    key = OBJECTIVE_TO_QUALITY_KEY[objective]
    return [float(item[key]) for item in row["candidate_quality"]]


def finite_mean(values):
    filtered = [float(v) for v in values if math.isfinite(float(v))]
    if not filtered:
        return float("inf")
    return sum(filtered) / len(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Select the best fixed/static same-budget mask from validation candidate-label losses."
    )
    parser.add_argument("--candidate_label_file", required=True)
    parser.add_argument("--candidate_metadata_file", default="")
    parser.add_argument("--objective", choices=["Delta_NLL", "KL"], default="Delta_NLL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    label_path = Path(args.candidate_label_file)
    meta_path = metadata_path_for(args.candidate_label_file, args.candidate_metadata_file)
    rows = load_rows(label_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    masks = list(metadata.get("candidate_masks") or [])
    if not masks:
        raise ValueError(f"Candidate metadata has no candidate_masks: {meta_path}")

    candidate_count = len(masks)
    losses_by_candidate = [[] for _ in range(candidate_count)]
    for row in rows:
        losses = loss_row(row, args.objective)
        if len(losses) != candidate_count:
            raise ValueError(
                f"Row sample_id={row.get('sample_id')} has {len(losses)} losses, expected {candidate_count}"
            )
        for idx, value in enumerate(losses):
            losses_by_candidate[idx].append(float(value))

    scores = []
    for idx, (mask, values) in enumerate(zip(masks, losses_by_candidate)):
        score = finite_mean(values)
        scores.append(
            {
                "candidate_index": idx,
                "mask_id": str(mask.get("mask_id")),
                "strategy": str(mask.get("strategy")),
                "mean_loss": score,
                "num_rows": len(values),
                "mask": [int(v) for v in mask["mask"]],
            }
        )

    best = min(scores, key=lambda item: (float(item["mean_loss"]), int(item["candidate_index"])))
    enriched_masks = []
    for mask, score in zip(masks, scores):
        row = dict(mask)
        row["metadata"] = dict(row.get("metadata") or {})
        row["metadata"].update(
            {
                "validation_objective": args.objective,
                "validation_mean_loss": float(score["mean_loss"]),
                "validation_rows": int(score["num_rows"]),
            }
        )
        enriched_masks.append(row)

    payload = {
        "version": 2,
        "schema": "opal_layer_masks",
        "source": "best_static_mask_from_validation_candidate_losses",
        "candidate_label_file": str(label_path),
        "candidate_metadata_file": str(meta_path),
        "objective": args.objective,
        "num_validation_rows": len(rows),
        "selected_index": int(best["candidate_index"]),
        "selected_mask_id": best["mask_id"],
        "selected_strategy": best["strategy"],
        "selected_mean_loss": float(best["mean_loss"]),
        "scores": scores,
        "masks": enriched_masks,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "Selected best static mask "
        f"{payload['selected_mask_id']} ({payload['selected_strategy']}) "
        f"with mean {args.objective}={payload['selected_mean_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
