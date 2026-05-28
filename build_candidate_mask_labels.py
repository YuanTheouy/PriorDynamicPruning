#!/usr/bin/env python3
import argparse
import json
import os
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
from opal_llm.quality_metrics import quality_rows_from_logits
from opal_llm.related_baselines import (
    DEFAULT_CANDIDATE_STRATEGIES,
    candidate_mask_specs,
    candidate_specs_to_dicts,
    deterministic_sample_indices,
    keep_count_from_args,
    load_risk_label_rows,
    parse_csv,
)


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


def forward_with_mask(model, input_ids, attention_mask, mask_payload=None):
    if mask_payload is None:
        clear_custom_policy(model)
    else:
        set_custom_policy(model, mask_payload)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    finally:
        clear_custom_policy(model)


def candidate_loss(quality, objective: str) -> float:
    if objective == "KL":
        return float(quality["KL_full_to_skip"])
    if objective == "Delta_NLL":
        return float(quality["Delta_NLL"])
    raise ValueError(f"Unsupported objective: {objective}")


def main():
    parser = argparse.ArgumentParser(description="Build PuDDing-style candidate-mask loss labels.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", default="")
    parser.add_argument("--calibration_file", default="")
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--risk_label_file", default="")
    parser.add_argument("--use_risk_label_sample_ids", action="store_true")
    parser.add_argument("--candidate_count", type=int, default=16)
    parser.add_argument("--candidate_strategies", default=",".join(DEFAULT_CANDIDATE_STRATEGIES))
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--top_k_layers", type=int, default=21)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--objective", choices=["KL", "Delta_NLL"], default="Delta_NLL")
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
    base_dataset = IndexedDataset(
        EvalSidDataset(
            train_file=data_file,
            tokenizer=tokenizer,
            category=dataset_category,
            max_len=2560,
            test=False,
            seed=args.seed,
        )
    )
    risk_rows = load_risk_label_rows(args.risk_label_file)
    allowed = risk_rows.keys() if args.use_risk_label_sample_ids and risk_rows else None
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    selected_indices = deterministic_sample_indices(
        total=len(base_dataset),
        max_samples=args.max_samples,
        seed=sample_seed,
        strategy=args.sample_strategy,
        allowed=allowed,
    )
    dataset = Subset(base_dataset, selected_indices)
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
    keep_count = keep_count_from_args(num_layers, args.skip_rate, args.top_k_layers)
    specs = candidate_mask_specs(
        num_layers=num_layers,
        top_k_layers=keep_count,
        seed=args.seed,
        candidate_count=args.candidate_count,
        risk_label_file=args.risk_label_file,
        strategies=parse_csv(args.candidate_strategies),
    )
    if not specs:
        raise ValueError("No candidate masks were generated.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")
    if shard_path.exists():
        shard_path.unlink()

    iterator = tqdm(dataloader, desc="candidate-labels") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sample_ids = [int(x) for x in batch["index"].detach().cpu().tolist()]
            full_logits = forward_with_mask(model, input_ids, attention_mask, None)
            row_losses = [[] for _ in sample_ids]
            row_quality = [[] for _ in sample_ids]

            for spec in specs:
                mask = torch.tensor([spec.mask] * input_ids.size(0), dtype=torch.float32, device=device)
                skip_logits = forward_with_mask(model, input_ids, attention_mask, mask)
                quality_rows = quality_rows_from_logits(full_logits, skip_logits, labels)
                for row_pos, quality in enumerate(quality_rows):
                    row_losses[row_pos].append(candidate_loss(quality, args.objective))
                    row_quality[row_pos].append(
                        {
                            "mask_id": spec.mask_id,
                            "Delta_NLL": float(quality["Delta_NLL"]),
                            "KL_full_to_skip": float(quality["KL_full_to_skip"]),
                            "NLL_full": float(quality["NLL_full"]),
                            "NLL_skip": float(quality["NLL_skip"]),
                        }
                    )

            for sample_id, losses, quality in zip(sample_ids, row_losses, row_quality):
                f.write(
                    json.dumps(
                        {
                            "sample_id": int(sample_id),
                            "objective": args.objective,
                            "num_layers": int(num_layers),
                            "skip_rate": float(args.skip_rate),
                            "top_k_layers": int(keep_count),
                            "candidate_losses": losses,
                            "candidate_quality": quality,
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
        metadata = {
            "teacher_model": args.teacher_model,
            "data_file": data_file,
            "category": args.category,
            "dataset_prompt": "EvalSidDataset",
            "objective": args.objective,
            "risk_label_file": args.risk_label_file,
            "used_risk_label_sample_ids": bool(args.use_risk_label_sample_ids and risk_rows),
            "candidate_count": len(specs),
            "candidate_masks": candidate_specs_to_dicts(specs),
            "num_layers": int(num_layers),
            "top_k_layers": int(keep_count),
            "skip_rate": float(args.skip_rate),
            "sample_strategy": args.sample_strategy,
            "sample_seed": int(sample_seed),
            "selected_indices": selected_indices,
            "num_samples": len(rows),
            "prompt_only_router_context": True,
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} candidate-label rows to {output_path}")
        print(f"Wrote candidate metadata to {metadata_path}")


if __name__ == "__main__":
    main()
