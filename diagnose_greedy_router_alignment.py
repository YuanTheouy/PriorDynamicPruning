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
from models.opal_risk_router import LayerQueryCrossAttentionRiskRouter, OpalRiskRouter
from opal_llm.quality_metrics import quality_rows_from_logits


CATEGORY_LABELS = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}
ATTENTION_ROUTER_INPUTS = {
    "prefix_hk_raw_attn",
    "prefix_hk_raw_attn_hk_last_resid",
    "prefix_hk_raw_attn_raw_hk_last_resid",
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


def prompt_only_inputs(input_ids, attention_mask, labels, pad_token_id):
    prompt_rows = []
    mask_rows = []
    lengths = []
    for ids, mask, row_labels in zip(input_ids, attention_mask, labels):
        valid_labels = row_labels.ne(-100).nonzero(as_tuple=False)
        prompt_len = int(valid_labels[0].item()) if valid_labels.numel() > 0 else int(mask.long().sum().item())
        prompt_len = max(1, min(prompt_len, ids.size(0)))
        prompt_rows.append(ids[:prompt_len])
        mask_rows.append(mask[:prompt_len])
        lengths.append(prompt_len)
    width = max(lengths)
    padded_ids = []
    padded_masks = []
    for ids, mask, length in zip(prompt_rows, mask_rows, lengths):
        pad = width - length
        if pad > 0:
            padded_ids.append(torch.cat([ids, ids.new_full((pad,), int(pad_token_id))], dim=0))
            padded_masks.append(torch.cat([mask, mask.new_zeros((pad,))], dim=0))
        else:
            padded_ids.append(ids)
            padded_masks.append(mask)
    return torch.stack(padded_ids, dim=0), torch.stack(padded_masks, dim=0)


def gather_last_token_state(last_hidden_state, attention_mask):
    rows = []
    for hidden, mask in zip(last_hidden_state, attention_mask):
        active = mask.ne(0).nonzero(as_tuple=False).flatten()
        rows.append(hidden[active[-1]] if active.numel() > 0 else hidden[-1])
    return torch.stack(rows, dim=0)


def masked_mean_state(hidden_state, attention_mask):
    weights = attention_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
    return (hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def pool_hidden_state(hidden_state, attention_mask, pooling: str):
    if pooling == "last":
        return gather_last_token_state(hidden_state, attention_mask)
    if pooling == "mean":
        return masked_mean_state(hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling: {pooling}")


def teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int):
    prefix_depth = max(0, min(int(prefix_depth), int(num_layers)))
    prefix_mask = [1 if idx < prefix_depth else 0 for idx in range(num_layers)]
    set_custom_policy(model, torch.tensor([prefix_mask] * input_ids.size(0), dtype=torch.float32, device=input_ids.device))
    try:
        with torch.no_grad():
            return model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    finally:
        clear_custom_policy(model)


def raw_embedding_hidden_state(model, input_ids):
    with torch.no_grad():
        return model.model.embed_tokens(input_ids)


def raw_embedding_state(model, input_ids, attention_mask):
    embeds = raw_embedding_hidden_state(model, input_ids)
    return masked_mean_state(embeds, attention_mask).float()


def load_router(path: str, hidden_size: int, num_layers: int, device):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    metadata = dict(checkpoint.get("metadata") or {})
    router_input = str(metadata.get("router_input") or "")
    router_architecture = str(metadata.get("router_architecture") or "")
    if router_input in ATTENTION_ROUTER_INPUTS or router_architecture == "layer_query_cross_attention":
        router = LayerQueryCrossAttentionRiskRouter(
            base_hidden_size=int(metadata.get("base_hidden_size") or hidden_size),
            num_layers=num_layers,
            router_dim=int(metadata.get("router_dim") or 256),
            router_heads=int(metadata.get("router_heads") or 4),
            use_hk_last_residual=bool(metadata.get("use_hk_last_residual")),
            use_raw_last_residual=bool(metadata.get("use_raw_last_residual")),
        )
    else:
        state_size = int(metadata.get("state_size") or metadata.get("hidden_size") or hidden_size)
        router = OpalRiskRouter(hidden_size=state_size, num_layers=num_layers)
    router.load_state_dict(state)
    router.to(device).eval()
    return router, metadata


def predict_risk(router, metadata, model, input_ids, attention_mask, labels, tokenizer, num_layers: int, default_prefix_depth: int):
    router_input = str(metadata.get("router_input") or "")
    prefix_depth = int(metadata.get("prefix_depth") or default_prefix_depth)
    router_input_ids, router_attention_mask = prompt_only_inputs(
        input_ids,
        attention_mask,
        labels,
        tokenizer.pad_token_id,
    )
    with torch.no_grad():
        if router_input == "raw_embedding":
            return router(raw_embedding_state(model, router_input_ids, router_attention_mask)).detach().float()[0]
        if router_input == "prefix_hk":
            hidden = teacher_prefix_hidden_state(model, router_input_ids, router_attention_mask, num_layers, prefix_depth)
            state = pool_hidden_state(hidden, router_attention_mask, str(metadata.get("risk_pooling") or "mean"))
            return router(state.float()).detach().float()[0]
        if router_input in ATTENTION_ROUTER_INPUTS:
            raw_hidden = raw_embedding_hidden_state(model, router_input_ids)
            hk_hidden = teacher_prefix_hidden_state(model, router_input_ids, router_attention_mask, num_layers, prefix_depth)
            return router(raw_hidden, hk_hidden, router_attention_mask).detach().float()[0]
    raise ValueError(f"Unsupported router_input for diagnostic: {router_input!r}")


def evaluate_kl(model, input_ids, attention_mask, labels, full_logits, num_layers: int, skipped_layers):
    mask = keep_mask_from_skipped(num_layers, skipped_layers, input_ids.device)
    skip_logits = forward_with_mask(model, input_ids, attention_mask, mask)
    quality = quality_rows_from_logits(full_logits, skip_logits, labels)[0]
    return {
        "KL_full_to_skip": float(quality["KL_full_to_skip"]),
        "Delta_NLL": float(quality["Delta_NLL"]),
        "Delta_PPL": float(quality["Delta_PPL"]),
    }


def summarize_rows(rows):
    if not rows:
        return {}

    def mean(key):
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    return {
        "num_samples": len(rows),
        "top1_match_rate": mean("top1_match"),
        "router_top1_in_greedy_first2_rate": mean("router_top1_in_greedy_first2"),
        "greedy_step1_in_router_top7_rate": mean("greedy_step1_in_router_top7"),
        "conditional_step2_match_rate": mean("conditional_step2_match"),
        "global_top2_match_conditional_step2_rate": mean("global_top2_match_conditional_step2"),
        "greedy_step2_in_router_top7_rate": mean("greedy_step2_in_router_top7"),
        "both_greedy_first2_in_router_top7_rate": mean("both_greedy_first2_in_router_top7"),
        "mean_step1_kl_gap_router_top1_minus_greedy": mean("step1_kl_gap_router_top1_minus_greedy"),
        "mean_step2_kl_gap_router_conditional_minus_greedy": mean("step2_kl_gap_router_conditional_minus_greedy"),
        "mean_router_top1_rank_in_single_kl": mean("router_top1_rank_in_single_kl"),
        "mean_greedy_step1_rank_in_router": mean("greedy_step1_rank_in_router"),
        "mean_greedy_step2_rank_in_router": mean("greedy_step2_rank_in_router"),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose two-step final-KL greedy choices vs router top-k ranking.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--risk_router_ckpt", required=True)
    parser.add_argument("--train_file", default="")
    parser.add_argument("--calibration_file", default="")
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--prefix_depth", type=int, default=4)
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--protected_head", type=int, default=4)
    parser.add_argument("--protected_tail", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="random")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
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
    hidden_size = int(model.config.hidden_size)
    router, router_metadata = load_router(args.risk_router_ckpt, hidden_size, num_layers, device)

    allowed_layers = list(range(max(0, int(args.protected_head)), max(0, int(num_layers) - max(0, int(args.protected_tail)))))
    if len(allowed_layers) < 2:
        raise ValueError("Need at least two allowed layers for two-step diagnostic.")

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
    if shard_path.exists():
        shard_path.unlink()

    iterator = tqdm(dataloader, desc="two-step-diagnostic") if accelerator.is_main_process else dataloader
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
                router_ranked = sorted(allowed_layers, key=lambda idx: float(pred_risk[int(idx)].item()))
                router_top1 = int(router_ranked[0])
                router_top2 = int(router_ranked[1])
                router_top7 = [int(idx) for idx in router_ranked[: min(7, len(router_ranked))]]

                single_quality = {}
                for layer_idx in allowed_layers:
                    single_quality[int(layer_idx)] = evaluate_kl(
                        model,
                        input_ids,
                        attention_mask,
                        labels,
                        full_logits,
                        num_layers,
                        [int(layer_idx)],
                    )
                greedy1 = min(single_quality, key=lambda idx: (single_quality[idx]["KL_full_to_skip"], idx))
                single_kl_ranked = sorted(allowed_layers, key=lambda idx: (single_quality[int(idx)]["KL_full_to_skip"], int(idx)))

                pair_quality = {}
                for layer_idx in allowed_layers:
                    if int(layer_idx) == int(greedy1):
                        continue
                    pair_quality[int(layer_idx)] = evaluate_kl(
                        model,
                        input_ids,
                        attention_mask,
                        labels,
                        full_logits,
                        num_layers,
                        [int(greedy1), int(layer_idx)],
                    )
                greedy2 = min(pair_quality, key=lambda idx: (pair_quality[idx]["KL_full_to_skip"], idx))
                router_after_greedy1 = next(int(idx) for idx in router_ranked if int(idx) != int(greedy1))
                global_top2_for_pair = router_top2 if router_top2 != greedy1 else int(router_ranked[2])

                row = {
                    "sample_id": int(sample_id),
                    "router_input": router_metadata.get("router_input"),
                    "risk_router_ckpt": args.risk_router_ckpt,
                    "skip_rate": float(args.skip_rate),
                    "num_layers": int(num_layers),
                    "allowed_layers": allowed_layers,
                    "router_top1": int(router_top1),
                    "router_top2": int(router_top2),
                    "router_top7": router_top7,
                    "greedy_step1": int(greedy1),
                    "greedy_step2_given_step1": int(greedy2),
                    "router_best_after_greedy_step1": int(router_after_greedy1),
                    "global_top2_for_pair": int(global_top2_for_pair),
                    "top1_match": int(router_top1 == greedy1),
                    "router_top1_in_greedy_first2": int(router_top1 in {int(greedy1), int(greedy2)}),
                    "greedy_step1_in_router_top7": int(greedy1 in router_top7),
                    "conditional_step2_match": int(router_after_greedy1 == greedy2),
                    "global_top2_match_conditional_step2": int(global_top2_for_pair == greedy2),
                    "greedy_step2_in_router_top7": int(greedy2 in router_top7),
                    "both_greedy_first2_in_router_top7": int(greedy1 in router_top7 and greedy2 in router_top7),
                    "step1_greedy_KL": float(single_quality[int(greedy1)]["KL_full_to_skip"]),
                    "step1_router_top1_KL": float(single_quality[int(router_top1)]["KL_full_to_skip"]),
                    "step1_kl_gap_router_top1_minus_greedy": float(
                        single_quality[int(router_top1)]["KL_full_to_skip"]
                        - single_quality[int(greedy1)]["KL_full_to_skip"]
                    ),
                    "step2_greedy_pair_KL": float(pair_quality[int(greedy2)]["KL_full_to_skip"]),
                    "step2_router_conditional_pair_KL": float(pair_quality[int(router_after_greedy1)]["KL_full_to_skip"]),
                    "step2_kl_gap_router_conditional_minus_greedy": float(
                        pair_quality[int(router_after_greedy1)]["KL_full_to_skip"]
                        - pair_quality[int(greedy2)]["KL_full_to_skip"]
                    ),
                    "router_top1_rank_in_single_kl": int(single_kl_ranked.index(router_top1) + 1),
                    "greedy_step1_rank_in_router": int(router_ranked.index(greedy1) + 1),
                    "greedy_step2_rank_in_router": int(router_ranked.index(greedy2) + 1),
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
                "data_file": data_file,
                "category": args.category,
                "risk_router_ckpt": args.risk_router_ckpt,
                "router_metadata": router_metadata,
                "protected_head": int(args.protected_head),
                "protected_tail": int(args.protected_tail),
                "skip_rate": float(args.skip_rate),
                "num_layers": int(num_layers),
                "allowed_layers": allowed_layers,
                "max_samples": int(args.max_samples),
                "sample_strategy": args.sample_strategy,
                "sample_seed": args.seed if args.sample_seed is None else args.sample_seed,
                "selected_indices": selected_indices,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Wrote diagnostic rows to {output_path}")
        print(f"Wrote diagnostic summary to {summary_path}")


if __name__ == "__main__":
    main()
