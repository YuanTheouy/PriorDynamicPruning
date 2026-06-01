#!/usr/bin/env python3
"""Readonly global OPAL loss/eval artifact audit.

This script does not launch training or eval. It only reads existing
training_metrics, summary CSV, and overlap summary JSON artifacts, then writes
a compact digest plus a full per-epoch Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRIC_KEYS = ("Delta_NLL", "Delta_PPL", "KL_full_to_skip", "NDCG@10", "retention_NDCG@10")
OVERLAP_KEYS = (
    "exact_match_rate",
    "mean_overlap_count",
    "mean_hamming_count",
    "mean_pairwise_order_accuracy",
    "unique_predicted_masks",
)
COMPONENT_KEYS = ("huber", "ranking", "skip_set")


@dataclass(frozen=True)
class AuditSpec:
    group: str
    name: str
    metric_globs: tuple[str, ...]
    eval_needles: tuple[str, ...] = ()
    eval_excludes: tuple[str, ...] = ()
    overlap_globs: tuple[str, ...] = ()
    required: bool = True


def specs() -> list[AuditSpec]:
    old_specs: list[AuditSpec] = []
    old_tasks = {
        "office25": "submission_office25_delta_m2000",
        "industrial25": "submission_industrial25_delta_m2000",
        "office36": "submission_office36_delta_m2000",
    }
    old_variants = {
        "raw_input_risk": "raw_input_risk",
        "prefix_hk_last": "opal_prefixlast",
        "prefix_hk_raw_attn": "opal_attn",
    }
    for task, run_group in old_tasks.items():
        for variant, eval_tag in old_variants.items():
            old_specs.append(
                AuditSpec(
                    group="old_one_layer_risk",
                    name=f"{task}:{variant}",
                    metric_globs=(
                        f"policy_ckpts/submission_convergence/{run_group}_seed42/{variant}/training_metrics.json",
                        f"policy_ckpts/final_opal_attn/final_opal_attn_delta_m2000_seed42/{variant}/training_metrics.json",
                        f"policy_ckpts/final_opal_attn_large/*delta*m2000*seed42*/{variant}/training_metrics.json",
                    ),
                    eval_needles=(f"{run_group}_seed42_{eval_tag}",),
                    required=True,
                )
            )

    p2_run_groups = (
        "delta_nll_greedy_set_m2000_seed42",
    )
    p2_exact_run_groups = (
        "delta_nll_greedy_set_exact_k_ce_m2000_seed42",
    )
    p2_eval_excludes = ("opal_setbce_", "opal_setattn_v1_")
    p2_specs = [
        AuditSpec(
            group="p2_delta_nll_greedy_set",
            name="raw_input_risk:bce",
            metric_globs=tuple(
                f"policy_ckpts/final_kl_greedy_set/{run}/raw_input_risk/training_metrics.json"
                for run in p2_run_groups
            ),
            eval_needles=tuple(f"{run}_raw_input_risk" for run in p2_run_groups),
            eval_excludes=p2_eval_excludes,
            overlap_globs=tuple(
                f"results/opal_greedy_set_diagnostics/{run}_raw_input_risk*_train_overlap.summary.json"
                for run in p2_run_groups
            ),
        ),
        AuditSpec(
            group="p2_delta_nll_greedy_set",
            name="prefix_hk_raw_attn:bce",
            metric_globs=tuple(
                f"policy_ckpts/final_kl_greedy_set/{run}/prefix_hk_raw_attn/training_metrics.json"
                for run in p2_run_groups
            ),
            eval_needles=tuple(f"{run}_prefix_hk_raw_attn" for run in p2_run_groups),
            eval_excludes=p2_eval_excludes,
            overlap_globs=tuple(
                f"results/opal_greedy_set_diagnostics/{run}_prefix_hk_raw_attn*_train_overlap.summary.json"
                for run in p2_run_groups
            ),
        ),
        AuditSpec(
            group="p2_delta_nll_greedy_set",
            name="raw_input_risk:exact_k_ce",
            metric_globs=tuple(
                f"policy_ckpts/final_kl_greedy_set_exact_k_ce/{run}/raw_input_risk_exact_k_ce/training_metrics.json"
                for run in p2_exact_run_groups
            ),
            eval_needles=tuple(f"{run}_raw_input_risk_exact_k_ce" for run in p2_exact_run_groups),
            eval_excludes=p2_eval_excludes,
            overlap_globs=tuple(
                f"results/opal_greedy_set_diagnostics/{run}_raw_input_risk_exact_k_ce_train_overlap.summary.json"
                for run in p2_exact_run_groups
            ),
        ),
        AuditSpec(
            group="p2_delta_nll_greedy_set",
            name="prefix_hk_last:exact_k_ce",
            metric_globs=tuple(
                f"policy_ckpts/final_kl_greedy_set_exact_k_ce/{run}/prefix_hk_last_exact_k_ce/training_metrics.json"
                for run in p2_exact_run_groups
            ),
            eval_needles=tuple(f"{run}_prefix_hk_last_exact_k_ce" for run in p2_exact_run_groups),
            eval_excludes=p2_eval_excludes,
            overlap_globs=tuple(
                f"results/opal_greedy_set_diagnostics/{run}_prefix_hk_last_exact_k_ce_train_overlap.summary.json"
                for run in p2_exact_run_groups
            ),
        ),
        AuditSpec(
            group="p2_delta_nll_greedy_set",
            name="prefix_hk_raw_attn:exact_k_ce",
            metric_globs=tuple(
                f"policy_ckpts/final_kl_greedy_set_exact_k_ce/{run}/prefix_hk_raw_attn_exact_k_ce/training_metrics.json"
                for run in p2_exact_run_groups
            ),
            eval_needles=tuple(f"{run}_prefix_hk_raw_attn_exact_k_ce" for run in p2_exact_run_groups),
            eval_excludes=p2_eval_excludes,
            overlap_globs=tuple(
                f"results/opal_greedy_set_diagnostics/{run}_prefix_hk_raw_attn_exact_k_ce_train_overlap.summary.json"
                for run in p2_exact_run_groups
            ),
        ),
    ]

    setattn_run = "opal_setattn_v1_delta_nll_greedy_set_m2000_seed42"
    setattn_specs = [
        AuditSpec(
            group="setattn_v1",
            name="prefix_hk_raw_setattn:bce",
            metric_globs=(
                f"policy_ckpts/opal_setattn_v1/{setattn_run}/prefix_hk_raw_setattn_bce/training_metrics.json",
                f"policy_ckpts/opal_setattn_v1_smoke/{setattn_run}/prefix_hk_raw_setattn_bce/training_metrics.json",
            ),
            eval_needles=(f"{setattn_run}_prefix_hk_raw_setattn_bce",),
            overlap_globs=(f"results/opal_greedy_set_diagnostics/{setattn_run}_prefix_hk_raw_setattn_bce_train_overlap.summary.json",),
        ),
        AuditSpec(
            group="setattn_v1",
            name="prefix_hk_raw_setattn:exact_k_ce",
            metric_globs=(
                f"policy_ckpts/opal_setattn_v1/{setattn_run}/prefix_hk_raw_setattn_exact_k_ce/training_metrics.json",
                f"policy_ckpts/opal_setattn_v1_smoke/{setattn_run}/prefix_hk_raw_setattn_exact_k_ce/training_metrics.json",
            ),
            eval_needles=(f"{setattn_run}_prefix_hk_raw_setattn_exact_k_ce",),
            overlap_globs=(f"results/opal_greedy_set_diagnostics/{setattn_run}_prefix_hk_raw_setattn_exact_k_ce_train_overlap.summary.json",),
        ),
    ]

    cross_specs = []
    for task in ("industrial25", "office36"):
        run = f"opal_setbce_{task}_delta_nll_greedy_set_m2000_seed42"
        cross_specs.append(
            AuditSpec(
                group="cross_setting_setbce",
                name=f"{task}:prefix_hk_raw_attn:bce",
                metric_globs=(
                    f"policy_ckpts/opal_setbce_cross_setting/{run}/prefix_hk_raw_attn_bce/training_metrics.json",
                    f"policy_ckpts/opal_setattn_v1_smoke/{run}/prefix_hk_raw_attn_bce/training_metrics.json",
                ),
                eval_needles=(f"{run}_prefix_hk_raw_attn_bce",),
                overlap_globs=(f"results/opal_greedy_set_diagnostics/{run}_prefix_hk_raw_attn_bce_train_overlap.summary.json",),
                required=False,
            )
        )

    return old_specs + p2_specs + setattn_specs + cross_specs


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value, digits: int = 4) -> str:
    number = safe_float(value)
    if math.isnan(number) or math.isinf(number):
        return "NA"
    return f"{number:.{digits}f}"


def pct(first, best) -> float:
    first_f = safe_float(first)
    best_f = safe_float(best)
    if math.isnan(first_f) or math.isnan(best_f):
        return math.nan
    return 100.0 * (first_f - best_f) / max(abs(first_f), 1e-12)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def first_existing(patterns: Iterable[str], base: Path) -> Path | None:
    for pattern in patterns:
        existing = sorted(Path(p) for p in glob.glob(str(base / pattern)) if Path(p).exists())
        if existing:
            return existing[0]
    return None


def all_existing(patterns: Iterable[str], base: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(base / pattern)))
    return sorted({path for path in matches if path.exists()})


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("history") or []
    finite = [(safe_float(row.get("loss")), row) for row in history]
    finite = [(loss, row) for loss, row in finite if not math.isnan(loss)]
    if not finite:
        return {"path": path, "history": history, "metadata": data.get("metadata") or {}}
    first = finite[0][1]
    best = min(finite, key=lambda item: item[0])[1]
    last = finite[-1][1]
    return {
        "path": path,
        "metadata": data.get("metadata") or {},
        "history": history,
        "first": first,
        "best": best,
        "last": last,
        "loss_drop_pct": pct(first.get("loss"), best.get("loss")),
        "skip_set_drop_pct": pct(first.get("skip_set"), best.get("skip_set")),
    }


def load_overlap(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": type(exc).__name__}
    data["path"] = path
    return data


def summary_csv_rows(base: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((base / "results/planrec_experiments/tables").glob("summary*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["_summary_path"] = path
                    rows.append(row)
        except Exception:
            continue
    return rows


def find_eval_rows(rows: list[dict], needles: tuple[str, ...], excludes: tuple[str, ...] = ()) -> list[dict]:
    if not needles:
        return []
    selected: list[dict] = []
    seen = set()
    for row in rows:
        run_name = row.get("run_name") or row.get("name") or ""
        text = " ".join(str(row.get(key, "")) for key in ("run_name", "json_path", "result_json", "_summary_path"))
        if excludes and any(exclude in text or exclude in run_name for exclude in excludes):
            continue
        if any(needle in text or needle in run_name for needle in needles):
            key = (str(row.get("_summary_path")), run_name)
            if key not in seen:
                seen.add(key)
                selected.append(row)
    return selected


def summarize_eval(row: dict) -> str:
    parts = []
    for key in METRIC_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={fmt(value, 4)}")
    return ", ".join(parts) if parts else "NA"


def summarize_overlap(row: dict | None) -> str:
    if row is None:
        return "NA"
    if row.get("error"):
        return f"ERROR:{row['error']}"
    parts = []
    labels = {
        "exact_match_rate": "exact",
        "mean_overlap_count": "overlap",
        "mean_hamming_count": "hamming",
        "mean_pairwise_order_accuracy": "pairwise",
        "unique_predicted_masks": "unique",
    }
    for key in OVERLAP_KEYS:
        if key in row:
            digits = 0 if key == "unique_predicted_masks" else 4
            parts.append(f"{labels[key]}={fmt(row.get(key), digits)}")
    return ", ".join(parts) if parts else "NA"


def component_triplet(metrics: dict | None, key: str) -> str:
    if not metrics or "first" not in metrics:
        return "NA"
    return "->".join(fmt(metrics[pos].get(key), 4) for pos in ("first", "best", "last"))


def loss_triplet(metrics: dict | None) -> str:
    if not metrics or "first" not in metrics:
        return "NA"
    drop = fmt(metrics.get("loss_drop_pct"), 1)
    return "{}->{}->{} (drop {}%)".format(
        fmt(metrics["first"].get("loss"), 4),
        fmt(metrics["best"].get("loss"), 4),
        fmt(metrics["last"].get("loss"), 4),
        drop,
    )


def short_path(path: Path | None, base: Path) -> str:
    return rel(path, base) if path else "MISSING"


def write_reports(base: Path, out_dir: Path, max_digest_sample_rows: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = summary_csv_rows(base)

    audit_rows = []
    for spec in specs():
        metric_path = first_existing(spec.metric_globs, base)
        metrics = load_metrics(metric_path) if metric_path else None
        overlap_paths = all_existing(spec.overlap_globs, base)
        overlap = load_overlap(overlap_paths[0] if overlap_paths else None)
        eval_matches = find_eval_rows(eval_rows, spec.eval_needles, spec.eval_excludes)
        status = "OK" if metric_path else ("PENDING" if not spec.required else "MISSING")
        if metric_path and not eval_matches:
            status += "+EVAL_MISSING"
        if spec.overlap_globs and not overlap_paths:
            status += "+OVERLAP_MISSING"
        audit_rows.append(
            {
                "spec": spec,
                "status": status,
                "metric_path": metric_path,
                "metrics": metrics,
                "overlap": overlap,
                "overlap_path": overlap_paths[0] if overlap_paths else None,
                "eval_rows": eval_matches,
            }
        )

    sample_metric_paths = sorted(
        {
            Path(path)
            for pattern in (
                "policy_ckpts/**/*m5000*/**/training_metrics.json",
                "policy_ckpts/**/*m10000*/**/training_metrics.json",
            )
            for path in glob.glob(str(base / pattern), recursive=True)
        }
    )
    sample_eval_rows = [
        row
        for row in eval_rows
        if any(tag in (row.get("run_name") or "") or tag in str(row.get("_summary_path")) for tag in ("m5000", "m10000"))
    ]

    digest_lines = [
        "# OPAL Global Loss Curve Audit Digest",
        "",
        "Readonly scan only: no training/eval launched.",
        "",
        "| group | run | status | training_metrics | epochs | loss first->best->last | skip_set first->best->last | overlap | eval |",
        "|---|---|---|---|---:|---|---|---|---|",
    ]
    for row in audit_rows:
        spec = row["spec"]
        metrics = row["metrics"]
        eval_summary = "NA"
        if row["eval_rows"]:
            eval_summary = " ; ".join(summarize_eval(item) for item in row["eval_rows"][:2])
            if len(row["eval_rows"]) > 2:
                eval_summary += f" ; +{len(row['eval_rows']) - 2} more"
        digest_lines.append(
            "| {group} | {name} | {status} | `{path}` | {epochs} | {loss} | {skip_set} | {overlap} | {eval} |".format(
                group=spec.group,
                name=spec.name,
                status=row["status"],
                path=short_path(row["metric_path"], base),
                epochs=len(metrics.get("history", [])) if metrics else 0,
                loss=loss_triplet(metrics),
                skip_set=component_triplet(metrics, "skip_set"),
                overlap=summarize_overlap(row["overlap"]),
                eval=eval_summary,
            )
        )

    digest_lines.extend(["", "## Sample Size Artifacts", ""])
    if sample_metric_paths or sample_eval_rows:
        digest_lines.append(f"training_metrics m5000/m10000 count: {len(sample_metric_paths)}")
        for path in sample_metric_paths[:max_digest_sample_rows]:
            digest_lines.append(f"- `{rel(path, base)}`")
        if len(sample_metric_paths) > max_digest_sample_rows:
            digest_lines.append(f"- ... +{len(sample_metric_paths) - max_digest_sample_rows} more")
        digest_lines.append(f"eval rows m5000/m10000 count: {len(sample_eval_rows)}")
        for row in sample_eval_rows[:max_digest_sample_rows]:
            digest_lines.append(f"- `{row.get('run_name', '')}`: {summarize_eval(row)}")
        if len(sample_eval_rows) > max_digest_sample_rows:
            digest_lines.append(f"- ... +{len(sample_eval_rows) - max_digest_sample_rows} more")
    else:
        digest_lines.append("PENDING: no m5000/m10000 training_metrics or eval summary rows found.")

    full_lines = [
        "# OPAL Global Loss Curve Audit Full",
        "",
        "Readonly scan only: no training/eval launched.",
        "",
    ]
    for row in audit_rows:
        spec = row["spec"]
        metrics = row["metrics"]
        full_lines.extend(
            [
                f"## {spec.group} / {spec.name}",
                "",
                f"- status: {row['status']}",
                f"- training_metrics: `{short_path(row['metric_path'], base)}`",
                f"- overlap_summary: `{short_path(row['overlap_path'], base)}`",
                f"- overlap: {summarize_overlap(row['overlap'])}",
            ]
        )
        if metrics and "first" in metrics:
            full_lines.extend(
                [
                    f"- epochs: {len(metrics.get('history', []))}",
                    f"- loss: {loss_triplet(metrics)}",
                    f"- huber: {component_triplet(metrics, 'huber')}",
                    f"- ranking: {component_triplet(metrics, 'ranking')}",
                    f"- skip_set: {component_triplet(metrics, 'skip_set')}",
                    "",
                    "| epoch | loss | huber | ranking | skip_set | steps |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for item in metrics.get("history", []):
                full_lines.append(
                    "| {epoch} | {loss} | {huber} | {ranking} | {skip_set} | {steps} |".format(
                        epoch=item.get("epoch", ""),
                        loss=fmt(item.get("loss"), 6),
                        huber=fmt(item.get("huber"), 6),
                        ranking=fmt(item.get("ranking"), 6),
                        skip_set=fmt(item.get("skip_set"), 6),
                        steps=item.get("steps", ""),
                    )
                )
        else:
            full_lines.append("")
        full_lines.extend(["", "### Eval Rows", ""])
        if row["eval_rows"]:
            full_lines.append("| summary_csv | run_name | Delta_NLL | Delta_PPL | KL | NDCG@10 | retention_NDCG@10 |")
            full_lines.append("|---|---|---:|---:|---:|---:|---:|")
            for eval_row in row["eval_rows"]:
                full_lines.append(
                    "| `{path}` | `{run}` | {dnll} | {dppl} | {kl} | {ndcg} | {ret} |".format(
                        path=rel(eval_row["_summary_path"], base),
                        run=eval_row.get("run_name", ""),
                        dnll=fmt(eval_row.get("Delta_NLL"), 6),
                        dppl=fmt(eval_row.get("Delta_PPL"), 6),
                        kl=fmt(eval_row.get("KL_full_to_skip"), 6),
                        ndcg=fmt(eval_row.get("NDCG@10"), 6),
                        ret=fmt(eval_row.get("retention_NDCG@10"), 6),
                    )
                )
        else:
            full_lines.append("MISSING/PENDING")
        full_lines.append("")

    full_lines.extend(["## Sample Size Artifacts", ""])
    if sample_metric_paths:
        full_lines.append("### m5000/m10000 training_metrics")
        for path in sample_metric_paths:
            try:
                metrics = load_metrics(path)
                full_lines.append(f"- `{rel(path, base)}`: {loss_triplet(metrics)}")
            except Exception as exc:  # noqa: BLE001
                full_lines.append(f"- `{rel(path, base)}`: ERROR {type(exc).__name__}")
    else:
        full_lines.append("No m5000/m10000 training_metrics found.")
    full_lines.append("")
    if sample_eval_rows:
        full_lines.append("### m5000/m10000 eval rows")
        for row in sample_eval_rows:
            full_lines.append(f"- `{row.get('run_name', '')}` ({rel(row['_summary_path'], base)}): {summarize_eval(row)}")
    else:
        full_lines.append("No m5000/m10000 eval summary rows found.")

    digest_path = out_dir / "opal_global_loss_curve_audit_digest.md"
    full_path = out_dir / "opal_global_loss_curve_audit_full.md"
    digest_path.write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
    full_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8")
    return digest_path, full_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Readonly OPAL global artifact audit.")
    parser.add_argument("--repo_dir", default=".", help="Repository root. Defaults to cwd.")
    parser.add_argument(
        "--output_dir",
        default="results/opal_global_loss_audit",
        help="Where to write digest/full audit Markdown files.",
    )
    parser.add_argument("--max_digest_sample_rows", type=int, default=12)
    parser.add_argument("--print", choices=("digest", "paths"), default="digest")
    args = parser.parse_args()

    base = Path(args.repo_dir).resolve()
    out_dir = (base / args.output_dir).resolve()
    digest_path, full_path = write_reports(base, out_dir, args.max_digest_sample_rows)

    print(f"DIGEST_PATH={digest_path}")
    print(f"FULL_PATH={full_path}")
    if args.print == "digest":
        print("=== DIGEST BEGIN ===")
        print(digest_path.read_text(encoding="utf-8"), end="")
        print("=== DIGEST END ===")


if __name__ == "__main__":
    main()
