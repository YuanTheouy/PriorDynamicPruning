#!/usr/bin/env python3
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

from transformers import AutoTokenizer, Qwen2ForCausalLM

from data import EvalSidDataset
from diagnose_greedy_router_alignment import (
    CATEGORY_LABELS,
    IndexedDataset,
    collate_batch,
    detect_num_layers,
    dtype_from_precision,
    load_router,
    predict_risk,
)
from opal_llm.mask_utils import mask_from_skip_risk


def read_label_rows(path: str, max_samples: int, sample_strategy: str, sample_seed: int):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows.sort(key=lambda row: int(row["sample_id"]))
    if max_samples > 0 and max_samples < len(rows):
        if sample_strategy == "random":
            rng = random.Random(int(sample_seed))
            rows = sorted(rng.sample(rows, int(max_samples)), key=lambda row: int(row["sample_id"]))
        else:
            rows = rows[: int(max_samples)]
    return rows


def pairwise_order_accuracy(pred_risk: torch.Tensor, target_skip_mask: torch.Tensor) -> float:
    skipped = pred_risk[target_skip_mask > 0.5]
    kept = pred_risk[target_skip_mask <= 0.5]
    if skipped.numel() == 0 or kept.numel() == 0:
        return 0.0
    # Lower risk should be skipped, so skipped layers should have lower risk than kept layers.
    return float((skipped[:, None] < kept[None, :]).float().mean().item())


def summarize_rows(rows):
    if not rows:
        return {}

    def mean(key):
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    pred_masks = collections.Counter(tuple(row["predicted_skipped_layers"]) for row in rows)
    label_masks = collections.Counter(tuple(row["label_skipped_layers"]) for row in rows)
    return {
        "num_samples": len(rows),
        "exact_match_rate": mean("exact_match"),
        "mean_overlap_count": mean("overlap_count"),
        "mean_overlap_ratio": mean("overlap_ratio"),
        "mean_hamming_count": mean("hamming_count"),
        "mean_hamming_ratio": mean("hamming_ratio"),
        "mean_pairwise_order_accuracy": mean("pairwise_order_accuracy"),
        "mean_label_skipped_risk": mean("label_skipped_risk_mean"),
        "mean_label_kept_risk": mean("label_kept_risk_mean"),
        "mean_keep_minus_skip_risk_margin": mean("keep_minus_skip_risk_margin"),
        "mean_predicted_skip_count": mean("predicted_skip_count"),
        "mean_label_skip_count": mean("label_skip_count"),
        "unique_predicted_masks": len(pred_masks),
        "unique_label_masks": len(label_masks),
        "top_predicted_masks": [
            {"mask": list(mask), "count": int(count), "rate": float(count / len(rows))}
            for mask, count in pred_masks.most_common(10)
        ],
        "top_label_masks": [
            {"mask": list(mask), "count": int(count), "rate": float(count / len(rows))}
            for mask, count in label_masks.most_common(10)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare router top-k skip masks with final-KL greedy skip-set labels.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--risk_router_ckpt", required=True)
    parser.add_argument("--risk_label_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--prefix_depth", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    label_rows = read_label_rows(args.risk_label_file, args.max_samples, args.sample_strategy, args.sample_seed)
    label_by_sample = {int(row["sample_id"]): row for row in label_rows}
    selected_indices = sorted(label_by_sample)
    if not selected_indices:
        raise ValueError(f"No label rows loaded from {args.risk_label_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_category = CATEGORY_LABELS.get(args.category, args.category)
    dataset = IndexedDataset(
        EvalSidDataset(
            train_file=args.train_file,
            tokenizer=tokenizer,
            category=dataset_category,
            max_len=2560,
            test=False,
            seed=args.seed,
        )
    )
    dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    dataloader = accelerator.prepare(dataloader)

    model = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype_from_precision(args.precision))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    router, router_metadata = load_router(args.risk_router_ckpt, hidden_size, num_layers, device)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        for stale_shard in output_path.parent.glob(output_path.name + ".rank*"):
            stale_shard.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")

    iterator = tqdm(dataloader, desc="greedy-set-overlap") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            sample_ids = [int(x) for x in batch["index"].detach().cpu().tolist()]

            for row_pos, sample_id in enumerate(sample_ids):
                label_row = label_by_sample[int(sample_id)]
                target_skip_mask = torch.tensor(label_row["skip_mask"], dtype=torch.float32, device=device)
                skip_count = int(target_skip_mask.sum().item())
                keep_count = int(num_layers) - skip_count

                input_ids = batch_input_ids[row_pos : row_pos + 1]
                attention_mask = batch_attention_mask[row_pos : row_pos + 1]
                labels = batch_labels[row_pos : row_pos + 1]
                pred_risk = predict_risk(
                    router,
                    router_metadata,
                    model,
                    input_ids,
                    attention_mask,
                    labels,
                    tokenizer,
                    num_layers,
                    args.prefix_depth,
                )
                keep_mask = mask_from_skip_risk(pred_risk.detach().float().cpu().tolist(), keep_count=keep_count)
                pred_skip_mask = torch.tensor([0 if int(value) == 1 else 1 for value in keep_mask], dtype=torch.float32)
                target_cpu = target_skip_mask.detach().cpu()

                predicted_skipped = [idx for idx, value in enumerate(pred_skip_mask.tolist()) if int(value) == 1]
                label_skipped = [idx for idx, value in enumerate(target_cpu.tolist()) if int(value) == 1]
                pred_set = set(predicted_skipped)
                label_set = set(label_skipped)
                overlap = len(pred_set & label_set)
                hamming = int((pred_skip_mask != target_cpu).sum().item())

                skipped_scores = pred_risk[target_skip_mask > 0.5]
                kept_scores = pred_risk[target_skip_mask <= 0.5]
                row = {
                    "sample_id": int(sample_id),
                    "router_input": router_metadata.get("router_input"),
                    "risk_router_ckpt": args.risk_router_ckpt,
                    "num_layers": int(num_layers),
                    "skip_count": int(skip_count),
                    "predicted_skipped_layers": predicted_skipped,
                    "label_skipped_layers": label_skipped,
                    "overlap_count": int(overlap),
                    "overlap_ratio": float(overlap / max(1, skip_count)),
                    "exact_match": int(pred_set == label_set),
                    "hamming_count": int(hamming),
                    "hamming_ratio": float(hamming / max(1, int(num_layers))),
                    "pairwise_order_accuracy": pairwise_order_accuracy(pred_risk, target_skip_mask),
                    "label_skipped_risk_mean": float(skipped_scores.float().mean().item()) if skipped_scores.numel() else 0.0,
                    "label_kept_risk_mean": float(kept_scores.float().mean().item()) if kept_scores.numel() else 0.0,
                    "keep_minus_skip_risk_margin": float(
                        kept_scores.float().mean().item() - skipped_scores.float().mean().item()
                    )
                    if skipped_scores.numel() and kept_scores.numel()
                    else 0.0,
                    "predicted_skip_count": int(len(predicted_skipped)),
                    "label_skip_count": int(len(label_skipped)),
                }
                f.write(json.dumps(row) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        rows = []
        for path in sorted(output_path.parent.glob(output_path.name + ".rank*")):
            with path.open(encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        rows.sort(key=lambda row: int(row["sample_id"]))
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = summarize_rows(rows)
        summary.update(
            {
                "teacher_model": args.teacher_model,
                "train_file": args.train_file,
                "risk_label_file": args.risk_label_file,
                "risk_router_ckpt": args.risk_router_ckpt,
                "router_metadata": router_metadata,
                "category": args.category,
                "max_samples": int(args.max_samples),
                "sample_strategy": args.sample_strategy,
                "sample_seed": int(args.sample_seed),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Wrote overlap rows to {output_path}")
        print(f"Wrote overlap summary to {summary_path}")


if __name__ == "__main__":
    main()
