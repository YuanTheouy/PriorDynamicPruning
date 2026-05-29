#!/usr/bin/env python3
import argparse
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
from opal_llm.mask_utils import keep_count_from_skip_rate
from opal_llm.quality_metrics import quality_rows_from_logits


CATEGORY_LABELS = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[index])
        row["index"] = index
        return row


def dtype_from_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def set_custom_policy(model, mask_payload):
    model.config.custom_layer_mask = mask_payload
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = mask_payload
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def clear_custom_policy(model):
    model.config.custom_layer_mask = None
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = None
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def collate_batch(batch, pad_token_id: int):
    batch = [row for row in batch if row is not None]
    input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in batch]
    attention_mask = [torch.tensor(row["attention_mask"], dtype=torch.long) for row in batch]
    labels = [torch.tensor(row["labels"], dtype=torch.long) for row in batch]
    indices = [int(row["index"]) for row in batch]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        "index": torch.tensor(indices, dtype=torch.long),
    }


def detect_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Could not detect decoder layer count.")


def forward_with_mask(model, input_ids, attention_mask, mask_payload=None):
    if mask_payload is None:
        clear_custom_policy(model)
    else:
        set_custom_policy(model, mask_payload)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    finally:
        clear_custom_policy(model)


def keep_mask_from_skipped(num_layers: int, skipped_layers, device):
    mask = torch.ones((1, int(num_layers)), dtype=torch.float32, device=device)
    for layer_idx in skipped_layers:
        mask[:, int(layer_idx)] = 0.0
    return mask


def greedy_final_kl_skip_set(
    model,
    input_ids,
    attention_mask,
    labels,
    full_logits,
    num_layers: int,
    allowed_layers,
    skip_count: int,
):
    selected = []
    steps = []
    for step_idx in range(int(skip_count)):
        best = None
        candidate_count = 0
        for layer_idx in allowed_layers:
            if layer_idx in selected:
                continue
            candidate_skips = selected + [int(layer_idx)]
            mask = keep_mask_from_skipped(num_layers, candidate_skips, input_ids.device)
            skip_logits = forward_with_mask(model, input_ids, attention_mask, mask)
            quality = quality_rows_from_logits(full_logits, skip_logits, labels)[0]
            candidate_kl = float(quality["KL_full_to_skip"])
            candidate_count += 1
            if best is None or candidate_kl < best["final_KL"] or (
                candidate_kl == best["final_KL"] and int(layer_idx) < int(best["layer"])
            ):
                best = {
                    "layer": int(layer_idx),
                    "final_KL": candidate_kl,
                    "Delta_NLL": float(quality["Delta_NLL"]),
                    "Delta_PPL": float(quality["Delta_PPL"]),
                }
        if best is None:
            break
        selected.append(int(best["layer"]))
        steps.append(
            {
                "step": step_idx + 1,
                "selected_layer": int(best["layer"]),
                "candidate_count": int(candidate_count),
                "final_KL": float(best["final_KL"]),
                "Delta_NLL": float(best["Delta_NLL"]),
                "Delta_PPL": float(best["Delta_PPL"]),
            }
        )
    skip_mask = [1 if idx in set(selected) else 0 for idx in range(int(num_layers))]
    keep_mask = [0 if value == 1 else 1 for value in skip_mask]
    return selected, skip_mask, keep_mask, steps


def main():
    parser = argparse.ArgumentParser(description="Build final-KL forward-greedy skip-set supervision labels.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", default="")
    parser.add_argument("--calibration_file", default="")
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--top_k_layers", type=int, default=21)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="random")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_file = args.calibration_file or args.train_file
    if not data_file:
        raise ValueError("Provide --train_file or --calibration_file.")

    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_category = CATEGORY_LABELS.get(args.category, args.category)
    dataset = IndexedDataset(
        EvalSidDataset(
            train_file=data_file,
            tokenizer=tokenizer,
            category=dataset_category,
            max_len=2560,
            test=False,
            seed=args.seed,
        )
    )
    selected_indices = list(range(len(dataset)))
    if args.max_samples > 0 and args.max_samples < len(selected_indices):
        if args.sample_strategy == "random":
            rng = random.Random(args.seed if args.sample_seed is None else args.sample_seed)
            selected_indices = sorted(rng.sample(selected_indices, int(args.max_samples)))
        else:
            selected_indices = selected_indices[: int(args.max_samples)]
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
    keep_count = keep_count_from_skip_rate(num_layers, args.skip_rate, args.top_k_layers)
    skip_count = int(num_layers) - int(keep_count)
    protected_head = max(0, int(args.protected_head))
    protected_tail = max(0, int(args.protected_tail))
    allowed_layers = list(range(protected_head, max(protected_head, int(num_layers) - protected_tail)))
    if skip_count > len(allowed_layers):
        raise ValueError(
            f"skip_count={skip_count} exceeds allowed layer count={len(allowed_layers)} "
            f"with protected_head={protected_head}, protected_tail={protected_tail}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")
    if shard_path.exists():
        shard_path.unlink()

    iterator = tqdm(dataloader, desc="greedy-set-labels") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            sample_ids = [int(x) for x in batch["index"].detach().cpu().tolist()]

            for row_pos, sample_id in enumerate(sample_ids):
                input_ids = batch_input_ids[row_pos : row_pos + 1]
                attention_mask = batch_attention_mask[row_pos : row_pos + 1]
                labels = batch_labels[row_pos : row_pos + 1]
                full_logits = forward_with_mask(model, input_ids, attention_mask, None)
                skipped_layers, skip_mask, keep_mask, steps = greedy_final_kl_skip_set(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    full_logits=full_logits,
                    num_layers=num_layers,
                    allowed_layers=allowed_layers,
                    skip_count=skip_count,
                )
                final_step = steps[-1] if steps else {"final_KL": 0.0, "Delta_NLL": 0.0, "Delta_PPL": 0.0}
                f.write(
                    json.dumps(
                        {
                            "sample_id": int(sample_id),
                            "objective": "final_KL",
                            "supervision_type": "skip_set",
                            "search": "forward_greedy",
                            "num_layers": int(num_layers),
                            "skip_rate": float(args.skip_rate),
                            "top_k_layers": int(keep_count),
                            "skip_count": int(skip_count),
                            "protected_head": int(protected_head),
                            "protected_tail": int(protected_tail),
                            "allowed_layers": allowed_layers,
                            "skipped_layers": skipped_layers,
                            "skip_mask": skip_mask,
                            "keep_mask": keep_mask,
                            "final_KL": float(final_step["final_KL"]),
                            "Delta_NLL": float(final_step["Delta_NLL"]),
                            "Delta_PPL": float(final_step["Delta_PPL"]),
                            "greedy_steps": steps,
                            "metadata": {"source_file": data_file},
                        }
                    )
                    + "\n"
                )

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
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "teacher_model": args.teacher_model,
                    "data_file": data_file,
                    "category": args.category,
                    "dataset_prompt": "EvalSidDataset",
                    "objective": "final_KL",
                    "supervision_type": "skip_set",
                    "search": "forward_greedy",
                    "skip_rate": float(args.skip_rate),
                    "top_k_layers": int(keep_count),
                    "skip_count": int(skip_count),
                    "protected_head": int(protected_head),
                    "protected_tail": int(protected_tail),
                    "allowed_layers": allowed_layers,
                    "sample_strategy": args.sample_strategy,
                    "sample_seed": args.seed if args.sample_seed is None else args.sample_seed,
                    "selected_indices": selected_indices,
                    "num_layers": int(num_layers),
                    "num_samples": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} final-KL greedy skip-set rows to {output_path}")
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
