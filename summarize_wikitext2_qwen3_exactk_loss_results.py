#!/usr/bin/env python3
import argparse
import glob
import json
import math
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def find_best_test(metric_dir: Path, prefix: str, best_epoch: int | None):
    if best_epoch is not None:
        matches = sorted(metric_dir.glob(f"{prefix}_best_val*_epoch{best_epoch:03d}_test.json"))
        if matches:
            return matches[0]
    matches = sorted(metric_dir.glob(f"{prefix}_best_val*_test.json"))
    return matches[0] if matches else None


def load_best_row(result_root: Path, run_id: str, prefix: str, method: str, loss: str):
    metric_dir = result_root / "val_ckpt_metrics" / run_id
    best_payload = load_json(metric_dir / "best_validation_checkpoint.json")
    if best_payload is None:
        return {
            "method": method,
            "loss": loss,
            "status": "missing",
            "run_id": run_id,
        }
    best = best_payload.get("best", {})
    epoch = int(best["epoch"]) if "epoch" in best else None
    test_path = find_best_test(metric_dir, prefix, epoch)
    test = load_json(test_path) if test_path else None
    return {
        "method": method,
        "loss": loss,
        "status": "ok" if test else "missing_test",
        "run_id": run_id,
        "best_epoch": epoch,
        "val_nll": best.get("nll"),
        "val_ppl": best.get("ppl"),
        "val_unique_masks": best.get("unique_masks"),
        "test_nll": None if test is None else test.get("nll"),
        "test_ppl": None if test is None else test.get("ppl"),
        "test_unique_masks": None if test is None else test.get("unique_masks"),
        "exact_skip_count_rate": None if test is None else test.get("exact_skip_count_rate"),
        "test_json": "" if test_path is None else str(test_path),
    }


def load_final_row(result_root: Path, label_run_id: str, filename: str, method: str, loss: str):
    path = result_root / "metrics" / label_run_id / filename
    payload = load_json(path)
    if payload is None:
        return {
            "method": method,
            "loss": loss,
            "status": "missing",
            "run_id": label_run_id,
        }
    return {
        "method": method,
        "loss": loss,
        "status": "ok",
        "run_id": label_run_id,
        "best_epoch": "final",
        "val_nll": None,
        "val_ppl": None,
        "val_unique_masks": None,
        "test_nll": payload.get("nll"),
        "test_ppl": payload.get("ppl"),
        "test_unique_masks": payload.get("unique_masks"),
        "exact_skip_count_rate": payload.get("exact_skip_count_rate"),
        "test_json": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/wikitext2_public_lm_sanity")
    parser.add_argument("--label_run_id", required=True)
    parser.add_argument("--raw_bce_run_id", default="")
    parser.add_argument("--opal_bce_run_id", default="")
    parser.add_argument("--raw_exactk_run_id", default="")
    parser.add_argument("--opal_exactk_run_id", default="")
    parser.add_argument("--output_md", default="docs/WIKITEXT2_QWEN3_EXACTK_LOSS_RESULTS.md")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    raw_bce = args.raw_bce_run_id or f"{args.label_run_id}_raw_valckpt"
    opal_bce = args.opal_bce_run_id or f"{args.label_run_id}_valckpt"
    raw_exactk = args.raw_exactk_run_id or f"{args.label_run_id}_raw_exactk_valckpt"
    opal_exactk = args.opal_exactk_run_id or f"{args.label_run_id}_opal_exactk_valckpt"

    rows = [
        load_final_row(result_root, args.label_run_id, "raw_setbce_test.json", "Raw final", "BCE"),
        load_final_row(result_root, args.label_run_id, "opal_setbce_test.json", "OPAL final", "BCE"),
        load_best_row(result_root, raw_bce, "raw", "Raw best-on-val", "BCE"),
        load_best_row(result_root, opal_bce, "opal", "OPAL best-on-val", "BCE"),
        load_best_row(result_root, raw_exactk, "raw", "Raw best-on-val", "Exact-K CE"),
        load_best_row(result_root, opal_exactk, "opal", "OPAL best-on-val", "Exact-K CE"),
    ]

    lines = [
        "# Qwen3-8B WikiText-2 Exact-K CE Loss Ablation",
        "",
        "This table reuses the existing clean Qwen3 K9 BCE artifacts and adds only the missing Exact-K CE rows.",
        "",
        f"- label run id: `{args.label_run_id}`",
        f"- Raw BCE val run id: `{raw_bce}`",
        f"- OPAL BCE val run id: `{opal_bce}`",
        f"- Raw Exact-K CE val run id: `{raw_exactk}`",
        f"- OPAL Exact-K CE val run id: `{opal_exactk}`",
        "",
        "| method | loss | epoch | val NLL | val PPL | val unique | test NLL | test PPL | test unique | exact-K | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {loss} | {epoch} | {val_nll} | {val_ppl} | {val_unique} | "
            "{test_nll} | {test_ppl} | {test_unique} | {exact} | {status} |".format(
                method=row["method"],
                loss=row["loss"],
                epoch=row.get("best_epoch", "NA"),
                val_nll=fmt(row.get("val_nll")),
                val_ppl=fmt(row.get("val_ppl")),
                val_unique=fmt(row.get("val_unique_masks"), 0),
                test_nll=fmt(row.get("test_nll")),
                test_ppl=fmt(row.get("test_ppl")),
                test_unique=fmt(row.get("test_unique_masks"), 0),
                exact=fmt(row.get("exact_skip_count_rate"), 3),
                status=row.get("status", "NA"),
            )
        )

    lines.extend(["", "## Source JSONs", ""])
    for row in rows:
        if row.get("test_json"):
            lines.append(f"- {row['method']} / {row['loss']}: `{row['test_json']}`")
    lines.append("")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
