#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from opal_llm.quality_metrics import summarize_quality_metrics
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
    parser.add_argument(
        "--run_name_contains",
        default="",
        help="Only include final JSON files whose metadata.run_name contains this substring.",
    )
    parser.add_argument(
        "--exclude_debug_sanity",
        action="store_true",
        help="Exclude rows marked or inferred as result_scope=debug_sanity.",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    raw_dir = root / "raw_json"
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / args.table_name
    md_path = table_dir / "quality_retention.md"
    md_header = "| dataset | run_name | scope | method | oracle_method | prompt_only | mask_id | stage | skip_rate | NDCG@10 | retention_NDCG@10 | Delta_NLL | Delta_PPL | KL_full_to_skip | oracle_regret | agreement | hamming | avg_layers | unique_masks |\n"
    md_separator = "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"

    rows = []
    payloads = []
    final_json_count = 0
    for path in sorted(raw_dir.glob("*.json")):
        if "_rank" in path.stem:
            continue
        final_json_count += 1
        payload = load_result(path)
        if payload is None:
            continue
        metadata = dict(payload.get("metadata") or {})
        run_name = str(metadata.get("run_name") or path.stem)
        if args.run_name_contains and args.run_name_contains not in run_name:
            continue
        if metadata.get("run_name") != run_name:
            payload = dict(payload)
            metadata["run_name"] = run_name
            payload["metadata"] = metadata
        payloads.append((path, payload))

    for path, payload in payloads:
        predictions = payload.get("predictions", [])
        quality_rows = [row for row in predictions if row.get("NLL_full") is not None]
        if quality_rows:
            payload = dict(payload)
            metrics = dict(payload.get("metrics") or {})
            metrics.update(summarize_quality_metrics(quality_rows))
            payload["metrics"] = metrics
        row = summary_row(payload, str(path))
        if args.exclude_debug_sanity and row.get("result_scope") == "debug_sanity":
            continue
        rows.append(row)

    if not rows:
        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["message", "raw_dir", "run_name_contains", "final_json_count"])
            writer.writeheader()
            writer.writerow(
                {
                    "message": "no matching final result JSON files",
                    "raw_dir": str(raw_dir),
                    "run_name_contains": args.run_name_contains,
                    "final_json_count": final_json_count,
                }
            )
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_header)
            f.write(md_separator)
        filter_msg = f" matching run_name_contains={args.run_name_contains!r}" if args.run_name_contains else ""
        print(f"No final result JSON files{filter_msg} found in {raw_dir}; scanned {final_json_count} final JSON files.")
        print(f"Wrote empty markdown table to {md_path}")
        return

    fields = list(rows[0].keys())
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_header)
        f.write(md_separator)
        for row in rows:
            f.write(
                "| {dataset} | {run_name} | {scope} | {method} | {oracle_method} | {prompt_only} | {mask_id} | {stage} | {skip_rate:.4f} | {ndcg10:.4f} | {retention:.4f} | {delta_nll:.6f} | {delta_ppl:.6f} | {kl:.6f} | {regret:.6f} | {agreement:.4f} | {hamming:.4f} | {layers:.2f} | {unique} |\n".format(
                    dataset=row.get("dataset"),
                    run_name=row.get("run_name") or "",
                    scope=row.get("result_scope") or "",
                    method=row.get("method"),
                    oracle_method=row.get("oracle_method") or "",
                    prompt_only=row.get("prompt_only_router_context")
                    if row.get("prompt_only_router_context") is not None
                    else "",
                    mask_id=row.get("mask_id") or "",
                    stage=row.get("opal_stage") or "",
                    skip_rate=float(row.get("skip_rate") or 0.0),
                    ndcg10=float(row.get("NDCG@10") or 0.0),
                    retention=float(row.get("retention_NDCG@10") or 0.0),
                    delta_nll=float(row.get("Delta_NLL") or 0.0),
                    delta_ppl=float(row.get("Delta_PPL") or 0.0),
                    kl=float(row.get("KL_full_to_skip") or 0.0),
                    regret=float(row.get("oracle_regret") or 0.0),
                    agreement=float(row.get("prefix_to_oracle_agreement") or 0.0),
                    hamming=float(row.get("hamming_distance") or 0.0),
                    layers=float(row.get("average_kept_layers") or 0.0),
                    unique=row.get("unique_masks"),
                )
            )

    print(f"Wrote {len(rows)} rows to {table_path}")
    print(f"Wrote markdown table to {md_path}")


if __name__ == "__main__":
    main()
