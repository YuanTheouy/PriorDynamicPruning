#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)
sys.path.insert(0, current_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext_opal_utils import (
    allowed_layers_from_policy,
    clear_custom_policy,
    collate_wikitext_windows,
    detect_num_layers,
    dtype_from_precision,
    keep_mask_from_skipped,
    lm_loss_stats_from_logits,
    load_wikitext_token_ids,
    resolve_skip_budget,
    set_custom_policy,
    WikitextWindowDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build WikiText-2 skip-set labels with final Delta_NLL objective."
    )
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_path", default="wikitext")
    parser.add_argument("--dataset_name", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_disk_path", default="")
    parser.add_argument("--dataset_cache_dir", default="")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--router_prefix_tokens", type=int, default=256)
    parser.add_argument("--label_samples", type=int, default=2000)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="random")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--skip_count", type=int, default=0)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=1)
    parser.add_argument("--search", choices=["forward_greedy", "beam"], default="forward_greedy")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def forward_nll_rows(model, input_ids, attention_mask, labels, layer_mask=None):
    if layer_mask is None:
        clear_custom_policy(model)
    else:
        set_custom_policy(model, layer_mask)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        rows = lm_loss_stats_from_logits(outputs.logits, labels)
        del outputs
        return rows
    finally:
        clear_custom_policy(model)


def evaluate_candidate_batch(
    model,
    input_ids,
    attention_mask,
    labels,
    num_layers: int,
    candidates,
):
    masks = []
    for skipped_layers in candidates:
        mask = torch.ones((num_layers,), dtype=torch.float32, device=input_ids.device)
        for layer_idx in skipped_layers:
            mask[int(layer_idx)] = 0.0
        masks.append(mask)
    layer_mask = torch.stack(masks, dim=0)
    expanded_input_ids = input_ids.expand(layer_mask.size(0), -1).contiguous()
    expanded_attention_mask = attention_mask.expand(layer_mask.size(0), -1).contiguous()
    expanded_labels = labels.expand(layer_mask.size(0), -1).contiguous()
    return forward_nll_rows(
        model,
        expanded_input_ids,
        expanded_attention_mask,
        expanded_labels,
        layer_mask=layer_mask,
    )


def greedy_skip_set(
    model,
    input_ids,
    attention_mask,
    labels,
    full_stats,
    num_layers: int,
    allowed_layers,
    skip_count: int,
    candidate_batch_size: int,
):
    selected = []
    steps = []
    full_nll = float(full_stats["nll"])
    full_ppl = float(full_stats["ppl"])
    for step_idx in range(int(skip_count)):
        candidate_payloads = []
        for layer_idx in allowed_layers:
            if int(layer_idx) in selected:
                continue
            candidate_payloads.append(selected + [int(layer_idx)])

        best = None
        for start in range(0, len(candidate_payloads), max(1, int(candidate_batch_size))):
            chunk = candidate_payloads[start : start + max(1, int(candidate_batch_size))]
            rows = evaluate_candidate_batch(
                model,
                input_ids,
                attention_mask,
                labels,
                num_layers=num_layers,
                candidates=chunk,
            )
            for skipped_layers, stats in zip(chunk, rows):
                candidate_layer = int(skipped_layers[-1])
                delta_nll = float(stats["nll"]) - full_nll
                delta_ppl = float(stats["ppl"]) - full_ppl
                row = {
                    "layer": candidate_layer,
                    "skipped_layers": [int(idx) for idx in skipped_layers],
                    "objective_loss": delta_nll,
                    "NLL_skip": float(stats["nll"]),
                    "PPL_skip": float(stats["ppl"]),
                    "Delta_NLL": delta_nll,
                    "Delta_PPL": delta_ppl,
                }
                if best is None or row["objective_loss"] < best["objective_loss"] or (
                    row["objective_loss"] == best["objective_loss"] and candidate_layer < int(best["layer"])
                ):
                    best = row
        if best is None:
            break
        selected.append(int(best["layer"]))
        steps.append(
            {
                "step": step_idx + 1,
                "selected_layer": int(best["layer"]),
                "candidate_count": int(len(candidate_payloads)),
                "objective": "Delta_NLL",
                "objective_loss": float(best["objective_loss"]),
                "NLL_skip": float(best["NLL_skip"]),
                "PPL_skip": float(best["PPL_skip"]),
                "Delta_NLL": float(best["Delta_NLL"]),
                "Delta_PPL": float(best["Delta_PPL"]),
            }
        )
    skip_set = set(selected)
    skip_mask = [1 if idx in skip_set else 0 for idx in range(int(num_layers))]
    keep_mask = [0 if value == 1 else 1 for value in skip_mask]
    return selected, skip_mask, keep_mask, steps


def beam_skip_set(
    model,
    input_ids,
    attention_mask,
    labels,
    full_stats,
    num_layers: int,
    allowed_layers,
    skip_count: int,
    candidate_batch_size: int,
    beam_width: int,
):
    beam_width = max(1, int(beam_width))
    full_nll = float(full_stats["nll"])
    full_ppl = float(full_stats["ppl"])
    beam = [
        {
            "skipped_layers": tuple(),
            "objective_loss": 0.0,
            "NLL_skip": full_nll,
            "PPL_skip": full_ppl,
            "Delta_NLL": 0.0,
            "Delta_PPL": 0.0,
        }
    ]
    steps = []

    for step_idx in range(int(skip_count)):
        candidate_payloads = []
        seen = set()
        for item in beam:
            selected = set(int(idx) for idx in item["skipped_layers"])
            for layer_idx in allowed_layers:
                layer_idx = int(layer_idx)
                if layer_idx in selected:
                    continue
                candidate = tuple(sorted(selected | {layer_idx}))
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidate_payloads.append(list(candidate))

        scored = []
        for start in range(0, len(candidate_payloads), max(1, int(candidate_batch_size))):
            chunk = candidate_payloads[start : start + max(1, int(candidate_batch_size))]
            rows = evaluate_candidate_batch(
                model,
                input_ids,
                attention_mask,
                labels,
                num_layers=num_layers,
                candidates=chunk,
            )
            for skipped_layers, stats in zip(chunk, rows):
                delta_nll = float(stats["nll"]) - full_nll
                delta_ppl = float(stats["ppl"]) - full_ppl
                scored.append(
                    {
                        "skipped_layers": tuple(int(idx) for idx in skipped_layers),
                        "objective_loss": delta_nll,
                        "NLL_skip": float(stats["nll"]),
                        "PPL_skip": float(stats["ppl"]),
                        "Delta_NLL": delta_nll,
                        "Delta_PPL": delta_ppl,
                    }
                )

        if not scored:
            break
        scored.sort(key=lambda row: (float(row["objective_loss"]), tuple(row["skipped_layers"])))
        beam = scored[:beam_width]
        best = beam[0]
        steps.append(
            {
                "step": step_idx + 1,
                "selected_layers": [int(idx) for idx in best["skipped_layers"]],
                "candidate_count": int(len(candidate_payloads)),
                "beam_width": int(beam_width),
                "kept_beam_count": int(len(beam)),
                "objective": "Delta_NLL",
                "objective_loss": float(best["objective_loss"]),
                "NLL_skip": float(best["NLL_skip"]),
                "PPL_skip": float(best["PPL_skip"]),
                "Delta_NLL": float(best["Delta_NLL"]),
                "Delta_PPL": float(best["Delta_PPL"]),
                "beam": [
                    {
                        "skipped_layers": [int(idx) for idx in item["skipped_layers"]],
                        "objective_loss": float(item["objective_loss"]),
                        "NLL_skip": float(item["NLL_skip"]),
                        "PPL_skip": float(item["PPL_skip"]),
                        "Delta_NLL": float(item["Delta_NLL"]),
                        "Delta_PPL": float(item["Delta_PPL"]),
                    }
                    for item in beam
                ],
            }
        )

    selected = [int(idx) for idx in beam[0]["skipped_layers"]] if beam else []
    skip_set = set(selected)
    skip_mask = [1 if idx in skip_set else 0 for idx in range(int(num_layers))]
    keep_mask = [0 if value == 1 else 1 for value in skip_mask]
    return selected, skip_mask, keep_mask, steps


def main():
    args = parse_args()
    if int(args.router_prefix_tokens) >= int(args.seq_len):
        raise ValueError("--router_prefix_tokens must be smaller than --seq_len")

    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    token_ids = load_wikitext_token_ids(
        tokenizer,
        split=args.split,
        dataset_disk_path=args.dataset_disk_path,
        dataset_cache_dir=args.dataset_cache_dir,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
    )
    dataset = WikitextWindowDataset(
        token_ids,
        seq_len=args.seq_len,
        max_windows=args.label_samples,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
    )
    selected_window_ids = list(dataset.window_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_wikitext_windows(batch, args.router_prefix_tokens),
    )
    dataloader = accelerator.prepare(dataloader)

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    skip_count, keep_count = resolve_skip_budget(num_layers, args.skip_rate, args.skip_count)
    allowed_layers = allowed_layers_from_policy(num_layers, args.protected_head, args.protected_tail)
    if skip_count > len(allowed_layers):
        raise ValueError(
            f"skip_count={skip_count} exceeds allowed layer count={len(allowed_layers)} "
            f"with protected_head={args.protected_head}, protected_tail={args.protected_tail}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_path.unlink(missing_ok=True)
        for stale in output_path.parent.glob(output_path.name + ".rank*"):
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")

    iterator = tqdm(dataloader, desc="wikitext-greedy-set") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            window_starts = [int(x) for x in batch["window_start"].detach().cpu().tolist()]

            for row_pos, sample_id in enumerate(sample_ids):
                input_ids = batch_input_ids[row_pos : row_pos + 1]
                attention_mask = batch_attention_mask[row_pos : row_pos + 1]
                labels = batch_labels[row_pos : row_pos + 1]
                full_stats = forward_nll_rows(model, input_ids, attention_mask, labels, layer_mask=None)[0]
                if args.search == "beam":
                    skipped_layers, skip_mask, keep_mask, steps = beam_skip_set(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        full_stats=full_stats,
                        num_layers=num_layers,
                        allowed_layers=allowed_layers,
                        skip_count=skip_count,
                        candidate_batch_size=args.candidate_batch_size,
                        beam_width=args.beam_width,
                    )
                    search_name = f"beam{max(1, int(args.beam_width))}"
                    step_key = "beam_steps"
                else:
                    skipped_layers, skip_mask, keep_mask, steps = greedy_skip_set(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        full_stats=full_stats,
                        num_layers=num_layers,
                        allowed_layers=allowed_layers,
                        skip_count=skip_count,
                        candidate_batch_size=args.candidate_batch_size,
                    )
                    search_name = "forward_greedy"
                    step_key = "greedy_steps"
                final_step = steps[-1] if steps else {
                    "objective_loss": 0.0,
                    "NLL_skip": float(full_stats["nll"]),
                    "PPL_skip": float(full_stats["ppl"]),
                    "Delta_NLL": 0.0,
                    "Delta_PPL": 0.0,
                }
                row = {
                    "sample_id": int(sample_id),
                    "window_start": int(window_starts[row_pos]),
                    "objective": "Delta_NLL",
                    "supervision_type": "skip_set",
                    "search": search_name,
                    "dataset": "wikitext-2-raw-v1",
                    "split": args.split,
                    "seq_len": int(args.seq_len),
                    "router_prefix_tokens": int(args.router_prefix_tokens),
                    "scored_tokens": int(full_stats["token_count"]),
                    "num_layers": int(num_layers),
                    "skip_rate": float(args.skip_rate),
                    "skip_count": int(skip_count),
                    "keep_count": int(keep_count),
                    "protected_head": int(args.protected_head),
                    "protected_tail": int(args.protected_tail),
                    "allowed_layers": [int(idx) for idx in allowed_layers],
                    "skipped_layers": [int(idx) for idx in skipped_layers],
                    "skip_mask": skip_mask,
                    "keep_mask": keep_mask,
                    "NLL_full": float(full_stats["nll"]),
                    "PPL_full": float(full_stats["ppl"]),
                    "NLL_skip": float(final_step["NLL_skip"]),
                    "PPL_skip": float(final_step["PPL_skip"]),
                    "objective_loss": float(final_step["objective_loss"]),
                    "Delta_NLL": float(final_step["Delta_NLL"]),
                    "Delta_PPL": float(final_step["Delta_PPL"]),
                    step_key: steps,
                    "metadata": {
                        "teacher_model": args.teacher_model,
                        "teacher_model_class": type(model).__name__,
                        "teacher_config_model_type": str(getattr(model.config, "model_type", "")),
                        "dataset_path": args.dataset_path,
                        "dataset_name": args.dataset_name,
                        "dataset_disk_path": args.dataset_disk_path,
                        "mask_application": "config.custom_layer_mask",
                        "target_leakage_guard": "router sees only first router_prefix_tokens; NLL scores suffix tokens only",
                    },
                }
                f.write(json.dumps(row) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        row_by_sample = {}
        for path in sorted(output_path.parent.glob(output_path.name + ".rank*")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_by_sample.setdefault(int(row["sample_id"]), row)
        rows = [row_by_sample[key] for key in sorted(row_by_sample)]
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        metadata = {
            "teacher_model": args.teacher_model,
            "teacher_model_class": type(model).__name__,
            "teacher_config_model_type": str(getattr(model.config, "model_type", "")),
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "dataset_disk_path": args.dataset_disk_path,
            "split": args.split,
            "num_tokens": int(len(token_ids)),
            "num_windows_available": int(len(dataset.starts)),
            "num_samples": int(len(rows)),
            "selected_window_ids": selected_window_ids,
            "sample_strategy": args.sample_strategy,
            "sample_seed": int(args.sample_seed),
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "scored_tokens_per_full_window": int(args.seq_len) - int(args.router_prefix_tokens),
            "objective": "Delta_NLL",
            "supervision_type": "skip_set",
            "search": f"beam{max(1, int(args.beam_width))}" if args.search == "beam" else "forward_greedy",
            "beam_width": int(max(1, int(args.beam_width))) if args.search == "beam" else 1,
            "num_layers": int(num_layers),
            "skip_rate": float(args.skip_rate),
            "skip_count": int(skip_count),
            "keep_count": int(keep_count),
            "protected_head": int(args.protected_head),
            "protected_tail": int(args.protected_tail),
            "allowed_layers": [int(idx) for idx in allowed_layers],
            "candidate_batch_size": int(args.candidate_batch_size),
            "mask_application": "config.custom_layer_mask",
            "target_leakage_guard": "router sees only first router_prefix_tokens; NLL scores suffix tokens only",
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} WikiText-2 Delta_NLL skip-set rows to {output_path}")
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
