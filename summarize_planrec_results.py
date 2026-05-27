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

    for path, payload in payloads:
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

    md_path = table_dir / "quality_retention.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| dataset | method | stage | skip_rate | NDCG@10 | retention_NDCG@10 | Delta_NLL | Delta_PPL | KL_full_to_skip | oracle_regret | agreement | hamming | avg_layers | unique_masks |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {dataset} | {method} | {stage} | {skip_rate:.4f} | {ndcg10:.4f} | {retention:.4f} | {delta_nll:.6f} | {delta_ppl:.6f} | {kl:.6f} | {regret:.6f} | {agreement:.4f} | {hamming:.4f} | {layers:.2f} | {unique} |\n".format(
                    dataset=row.get("dataset"),
                    method=row.get("method"),
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
