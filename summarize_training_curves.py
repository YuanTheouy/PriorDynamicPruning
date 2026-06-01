#!/usr/bin/env python3
"""Summarize OPAL/router training_metrics.json files.

The script is intentionally dependency-free so it can run on the server while
long submission jobs are still running.
"""

import argparse
import csv
import json
import math
from pathlib import Path


COMPONENT_KEYS = ("huber", "ranking", "skip_set")


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value, digits=6):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def rel_drop(first, best):
    if first is None or best is None or math.isnan(first) or math.isnan(best):
        return math.nan
    denom = max(abs(first), 1e-12)
    return 100.0 * (first - best) / denom


def flatten_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    keep = [
        "router_input",
        "supervision_type",
        "risk_label_rows",
        "candidate_label_rows",
        "covered_rows",
        "skip_count",
        "keep_count",
        "ranking_loss_weight",
        "skip_set_loss_weight",
        "set_loss_type",
        "num_candidates",
        "baseline",
    ]
    return {key: metadata.get(key, "") for key in keep}


def summarize_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") or []
    metadata = payload.get("metadata") or {}
    if not history:
        return None

    rows = []
    for item in history:
        loss = safe_float(item.get("loss"))
        epoch = item.get("epoch", "")
        rows.append((loss, epoch, item))
    finite_rows = [row for row in rows if not math.isnan(row[0])]
    if not finite_rows:
        return None

    first_loss, first_epoch, first_item = finite_rows[0]
    best_loss, best_epoch, best_item = min(finite_rows, key=lambda row: row[0])
    last_loss, last_epoch, last_item = finite_rows[-1]
    drop_pct = rel_drop(first_loss, best_loss)
    last_gap_pct = 100.0 * (last_loss - best_loss) / max(abs(best_loss), 1e-12)

    flags = []
    if drop_pct < 1.0:
        flags.append("几乎不降<1%")
    elif drop_pct < 5.0:
        flags.append("下降很小<5%")
    if last_gap_pct > 20.0:
        flags.append("末轮明显回升>20%")
    if any(math.isnan(row[0]) or math.isinf(row[0]) for row in rows):
        flags.append("有NaN/Inf")

    result = {
        "path": str(path),
        "run_dir": path.parent.name,
        "group_dir": path.parent.parent.name,
        "epochs": len(history),
        "first_epoch": first_epoch,
        "first_loss": first_loss,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "last_epoch": last_epoch,
        "last_loss": last_loss,
        "drop_pct": drop_pct,
        "last_gap_pct": last_gap_pct,
        "flags": ",".join(flags),
    }
    for prefix, item in (("first", first_item), ("best", best_item), ("last", last_item)):
        for key in COMPONENT_KEYS:
            result[f"{prefix}_{key}"] = safe_float(item.get(key))
    result.update(flatten_metadata(metadata))
    return result


def collect(paths, contains):
    seen = set()
    rows = []
    for root in paths:
        root_path = Path(root)
        if root_path.is_file() and root_path.name == "training_metrics.json":
            candidates = [root_path]
        elif root_path.exists():
            candidates = sorted(root_path.rglob("training_metrics.json"))
        else:
            continue
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if contains and contains not in str(path):
                continue
            try:
                summary = summarize_metrics(path)
            except Exception as exc:  # noqa: BLE001
                summary = {
                    "path": str(path),
                    "run_dir": path.parent.name,
                    "group_dir": path.parent.parent.name,
                    "epochs": 0,
                    "flags": f"读取失败:{type(exc).__name__}",
                }
            if summary:
                rows.append(summary)
    return sorted(rows, key=lambda row: (row.get("group_dir", ""), row.get("run_dir", ""), row.get("path", "")))


def write_csv(rows, output):
    if not rows:
        Path(output).write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output):
    lines = [
        "# Training Loss Curve Audit",
        "",
        "| group | variant | epochs | first | best | last | drop% | flags | huber first/best/last | ranking first/best/last | skip_set first/best/last | path |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        huber = "/".join(fmt(row.get(f"{pos}_huber"), 4) for pos in ("first", "best", "last"))
        ranking = "/".join(fmt(row.get(f"{pos}_ranking"), 4) for pos in ("first", "best", "last"))
        skip_set = "/".join(fmt(row.get(f"{pos}_skip_set"), 4) for pos in ("first", "best", "last"))
        lines.append(
            "| {group} | {variant} | {epochs} | {first} | {best} | {last} | {drop} | {flags} | {huber} | {ranking} | {skip_set} | `{path}` |".format(
                group=row.get("group_dir", ""),
                variant=row.get("run_dir", ""),
                epochs=row.get("epochs", ""),
                first=fmt(row.get("first_loss"), 4),
                best=fmt(row.get("best_loss"), 4),
                last=fmt(row.get("last_loss"), 4),
                drop=fmt(row.get("drop_pct"), 1),
                flags=row.get("flags", ""),
                huber=huber,
                ranking=ranking,
                skip_set=skip_set,
                path=row.get("path", ""),
            )
        )
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize router training loss curves.")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory or one training_metrics.json file. Can be repeated.",
    )
    parser.add_argument("--contains", default="", help="Only include paths containing this substring.")
    parser.add_argument("--output_csv", default="", help="Optional CSV output path.")
    parser.add_argument("--output_md", default="", help="Optional Markdown output path.")
    parser.add_argument("--limit", type=int, default=0, help="Print at most this many rows to stdout.")
    args = parser.parse_args()

    roots = args.root or [
        "policy_ckpts",
        "results",
    ]
    rows = collect(roots, args.contains)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    if args.output_md:
        write_markdown(rows, args.output_md)

    shown = rows if args.limit <= 0 else rows[: args.limit]
    print("group\tvariant\tepochs\tfirst\tbest\tlast\tdrop%\tflags\tpath")
    for row in shown:
        print(
            "\t".join(
                [
                    str(row.get("group_dir", "")),
                    str(row.get("run_dir", "")),
                    str(row.get("epochs", "")),
                    fmt(row.get("first_loss"), 6),
                    fmt(row.get("best_loss"), 6),
                    fmt(row.get("last_loss"), 6),
                    fmt(row.get("drop_pct"), 2),
                    str(row.get("flags", "")),
                    str(row.get("path", "")),
                ]
            )
        )
    if args.limit > 0 and len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more rows")


if __name__ == "__main__":
    main()
