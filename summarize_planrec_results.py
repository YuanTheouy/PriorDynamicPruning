#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from opal_llm.results import summary_row


def load_result(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metadata" not in payload or "metrics" not in payload:
        return None
    return payload


def main():
    parser = argparse.ArgumentParser(description="Summarize OPAL-LLM MiniOneRec result JSON files.")
    parser.add_argument("--output_dir", default="./results/planrec_experiments")
    parser.add_argument("--table_name", default="summary_recomputed.csv")
    args = parser.parse_args()

    root = Path(args.output_dir)
    raw_dir = root / "raw_json"
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    payloads = []
    for path in sorted(raw_dir.glob("*.json")):
        if "_rank" in path.stem:
            continue
        payload = load_result(path)
        if payload is None:
            continue
        payloads.append((path, payload))

    full_latency = {}
    for path, payload in payloads:
        metadata = payload.get("metadata", {})
        if metadata.get("method") != "full":
            continue
        key = (
            metadata.get("dataset"),
            metadata.get("beam_size"),
            metadata.get("max_new_tokens"),
            metadata.get("batch_size"),
            metadata.get("precision"),
        )
        latency = payload.get("latency", {}).get("latency_mean_sec")
        if latency:
            full_latency[key] = float(latency)

    for path, payload in payloads:
        metadata = payload.get("metadata", {})
        key = (
            metadata.get("dataset"),
            metadata.get("beam_size"),
            metadata.get("max_new_tokens"),
            metadata.get("batch_size"),
            metadata.get("precision"),
        )
        current = payload.get("latency", {}).get("latency_mean_sec")
        baseline = full_latency.get(key)
        if current and baseline:
            payload.setdefault("metadata", {})["speedup_vs_full"] = baseline / float(current)
        rows.append(summary_row(payload, str(path)))

    if not rows:
        print(f"No final result JSON files found in {raw_dir}")
        return

    table_path = table_dir / args.table_name
    fields = list(rows[0].keys())
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = table_dir / "quality_latency.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| dataset | method | budget | NDCG@10 | HR@10 | latency_mean_sec | speedup_vs_full | avg_layers | unique_masks |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {dataset} | {method} | {budget} | {ndcg10:.4f} | {hr10:.4f} | {lat:.4f} | {speedup} | {layers:.2f} | {unique} |\n".format(
                    dataset=row.get("dataset"),
                    method=row.get("method"),
                    budget=row.get("budget"),
                    ndcg10=float(row.get("NDCG@10") or 0.0),
                    hr10=float(row.get("HR@10") or 0.0),
                    lat=float(row.get("latency_mean_sec") or 0.0),
                    speedup=("" if row.get("speedup_vs_full") in (None, "") else f"{float(row.get('speedup_vs_full')):.3f}"),
                    layers=float(row.get("average_kept_layers") or 0.0),
                    unique=row.get("unique_masks"),
                )
            )

    print(f"Wrote {len(rows)} rows to {table_path}")
    print(f"Wrote markdown table to {md_path}")


if __name__ == "__main__":
    main()
