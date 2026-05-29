#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

from transformers import AutoTokenizer, Qwen2ForCausalLM

from data import EvalSidDataset
from models.opal_risk_router import (
    LayerQueryCrossAttentionRiskRouter,
    OpalRiskRouter,
    exact_k_subset_ce_loss,
    risk_pairwise_ranking_loss,
    risk_regression_loss,
    risk_skip_set_loss,
)
from opal_llm.mask_utils import allowed_layers_from_protected, keep_count_from_skip_rate


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


def recency_weighted_state(hidden_state, attention_mask, recent_tokens: int, recent_decay: float):
    recent_tokens = max(1, int(recent_tokens))
    recent_decay = float(recent_decay)
    rows = []
    for hidden, mask in zip(hidden_state, attention_mask):
        active = mask.ne(0).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            rows.append(hidden[-1])
        else:
            recent = hidden.index_select(0, active[-recent_tokens:])
            powers = torch.arange(recent.size(0) - 1, -1, -1, device=hidden.device, dtype=torch.float32)
            weights = torch.pow(torch.tensor(recent_decay, device=hidden.device, dtype=torch.float32), powers)
            weights = (weights / weights.sum().clamp_min(1e-12)).to(dtype=recent.dtype).unsqueeze(-1)
            rows.append((recent * weights).sum(dim=0))
    return torch.stack(rows, dim=0)


def pool_hidden_state(hidden_state, attention_mask, pooling: str):
    if pooling == "last":
        return gather_last_token_state(hidden_state, attention_mask)
    if pooling == "mean":
        return masked_mean_state(hidden_state, attention_mask)
    raise ValueError(f"Unsupported risk pooling: {pooling}")


def teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int):
    prefix_depth = max(0, min(int(prefix_depth), int(num_layers)))
    prefix_mask = [1 if idx < prefix_depth else 0 for idx in range(num_layers)]
    set_custom_policy(model, torch.tensor([prefix_mask] * input_ids.size(0), dtype=torch.float32, device=input_ids.device))
    try:
        with torch.no_grad():
            return model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    finally:
        clear_custom_policy(model)


def teacher_prefix_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int, pooling: str):
    hidden = teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers, prefix_depth)
    return pool_hidden_state(hidden, attention_mask, pooling).float()


def raw_embedding_hidden_state(model, input_ids):
    with torch.no_grad():
        return model.model.embed_tokens(input_ids)


def raw_embedding_state(model, input_ids, attention_mask):
    embeds = raw_embedding_hidden_state(model, input_ids)
    return masked_mean_state(embeds, attention_mask).float()


def fusion_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int, recent_tokens: int, recent_decay: float):
    raw_hidden = raw_embedding_hidden_state(model, input_ids)
    hk_hidden = teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers, prefix_depth)
    features = [
        gather_last_token_state(raw_hidden, attention_mask),
        recency_weighted_state(raw_hidden, attention_mask, recent_tokens, recent_decay),
        gather_last_token_state(hk_hidden, attention_mask),
        recency_weighted_state(hk_hidden, attention_mask, recent_tokens, recent_decay),
    ]
    return torch.cat(features, dim=-1).float()


def last_fusion_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int):
    raw_hidden = raw_embedding_hidden_state(model, input_ids)
    hk_hidden = teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers, prefix_depth)
    return torch.cat(
        [
            gather_last_token_state(raw_hidden, attention_mask),
            gather_last_token_state(hk_hidden, attention_mask),
        ],
        dim=-1,
    ).float()


def router_state_size(hidden_size: int, router_input: str) -> int:
    if router_input in ATTENTION_ROUTER_INPUTS:
        return int(hidden_size)
    if router_input == "prefix_hk_raw_fusion":
        return int(hidden_size) * 4
    if router_input == "prefix_hk_raw_last":
        return int(hidden_size) * 2
    return int(hidden_size)


def router_prefix_depth(router_input: str, prefix_depth: int) -> int:
    return int(prefix_depth) if router_input in {"prefix_hk", "prefix_hk_raw_fusion", "prefix_hk_raw_last"} or router_input in ATTENTION_ROUTER_INPUTS else 0


def router_pooling_metadata(router_input: str, risk_pooling: str) -> str:
    if router_input == "prefix_hk":
        return risk_pooling
    if router_input in ATTENTION_ROUTER_INPUTS:
        return "layer_query_cross_attention"
    if router_input == "prefix_hk_raw_fusion":
        return "raw_last+raw_recent_weighted+hk_last+hk_recent_weighted"
    if router_input == "prefix_hk_raw_last":
        return "raw_last+hk_last"
    return "mean"


def router_features_metadata(router_input: str):
    if router_input in ATTENTION_ROUTER_INPUTS:
        features = ["raw_token_sequence", "hk_token_sequence", "attention_mask", "layer_queries"]
        if router_input in {"prefix_hk_raw_attn_hk_last_resid", "prefix_hk_raw_attn_raw_hk_last_resid"}:
            features.append("hk_last_residual")
        if router_input == "prefix_hk_raw_attn_raw_hk_last_resid":
            features.append("raw_last_residual")
        return features
    if router_input == "prefix_hk_raw_fusion":
        return ["raw_last", "raw_recent_weighted", "hk_last", "hk_recent_weighted"]
    if router_input == "prefix_hk_raw_last":
        return ["raw_last", "hk_last"]
    if router_input == "prefix_hk":
        return ["hk_pooled"]
    return ["raw_embedding_mean"]


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


def load_supervision_labels(path: str):
    labels = {}
    metadata = {}
    supervision_type = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "skip_mask" in row:
                row_type = "skip_set"
                labels[int(row["sample_id"])] = [float(v) for v in row["skip_mask"]]
            else:
                row_type = "risk_regression"
                labels[int(row["sample_id"])] = [float(v) for v in row["risk_labels"]]
            if supervision_type is None:
                supervision_type = row_type
            elif supervision_type != row_type:
                raise ValueError(f"Mixed supervision label types in {path}: {supervision_type} and {row_type}")
            metadata.setdefault("supervision_type", row.get("supervision_type") or row_type)
            for key in [
                "objective",
                "num_layers",
                "prefix_depth",
                "search",
                "protected_head",
                "protected_tail",
                "skip_count",
                "allowed_layers",
            ]:
                value = row.get(key)
                if value is not None and metadata.get(key) is None:
                    metadata[key] = value
    if not labels:
        raise ValueError(f"No supervision labels found in {path}")
    metadata["supervision_type"] = supervision_type or metadata.get("supervision_type") or "risk_regression"
    return labels, metadata


def main():
    parser = argparse.ArgumentParser(description="Train prefix-supervised layer risk router.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--risk_label_file", required=True)
    parser.add_argument(
        "--router_input",
        choices=[
            "prefix_hk",
            "raw_embedding",
            "prefix_hk_raw_fusion",
            "prefix_hk_raw_last",
            "prefix_hk_raw_attn",
            "prefix_hk_raw_attn_hk_last_resid",
            "prefix_hk_raw_attn_raw_hk_last_resid",
        ],
        default="prefix_hk",
    )
    parser.add_argument("--prefix_depth", type=int, default=4)
    parser.add_argument("--risk_pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--recent_tokens", type=int, default=32)
    parser.add_argument("--recent_decay", type=float, default=0.85)
    parser.add_argument("--router_dim", type=int, default=256)
    parser.add_argument("--router_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--skip_set_loss_weight", type=float, default=0.0)
    parser.add_argument("--set_loss_type", choices=["bce", "exact_k_ce"], default="bce")
    parser.add_argument("--skip_rate", type=float, default=-1.0)
    parser.add_argument("--top_k_layers", type=int, default=0)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--loss_log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    risk_labels, label_metadata = load_supervision_labels(args.risk_label_file)
    supervision_type = str(label_metadata.get("supervision_type") or "risk_regression")
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
    covered_indices = sorted(index for index in risk_labels if 0 <= index < len(dataset))
    if not covered_indices:
        raise ValueError("No risk-label sample_id values match the training dataset indices.")
    dataset = Subset(dataset, covered_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    model = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype_from_precision(args.precision))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    state_size = router_state_size(hidden_size, args.router_input)
    keep_count = keep_count_from_skip_rate(
        num_layers,
        args.skip_rate,
        args.top_k_layers if args.top_k_layers > 0 else num_layers,
    )
    skip_count = num_layers - keep_count
    protected_head = label_metadata.get("protected_head")
    protected_tail = label_metadata.get("protected_tail")
    allowed_layers = label_metadata.get("allowed_layers")
    if not allowed_layers and (protected_head is not None or protected_tail is not None):
        allowed_layers = allowed_layers_from_protected(num_layers, protected_head, protected_tail)
    if not allowed_layers:
        allowed_layers = list(range(num_layers))
    if args.router_input in ATTENTION_ROUTER_INPUTS:
        router = LayerQueryCrossAttentionRiskRouter(
            base_hidden_size=hidden_size,
            num_layers=num_layers,
            router_dim=args.router_dim,
            router_heads=args.router_heads,
            dropout=args.dropout,
            use_hk_last_residual=args.router_input in {
                "prefix_hk_raw_attn_hk_last_resid",
                "prefix_hk_raw_attn_raw_hk_last_resid",
            },
            use_raw_last_residual=args.router_input == "prefix_hk_raw_attn_raw_hk_last_resid",
        ).to(device)
        router_architecture = "layer_query_cross_attention"
    else:
        router = OpalRiskRouter(hidden_size=state_size, num_layers=num_layers, dropout=args.dropout).to(device)
        router_architecture = "pooled_mlp"
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "router_input": args.router_input,
                    "risk_pooling": router_pooling_metadata(args.router_input, args.risk_pooling),
                    "router_features": router_features_metadata(args.router_input),
                    "dataset_prompt": "EvalSidDataset",
                    "num_layers": num_layers,
                    "base_hidden_size": hidden_size,
                    "state_size": state_size,
                    "router_architecture": router_architecture,
                    "router_dim": int(args.router_dim) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
                    "router_heads": int(args.router_heads) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
                    "use_hk_last_residual": args.router_input in {
                        "prefix_hk_raw_attn_hk_last_resid",
                        "prefix_hk_raw_attn_raw_hk_last_resid",
                    },
                    "use_raw_last_residual": args.router_input == "prefix_hk_raw_attn_raw_hk_last_resid",
                    "recent_tokens": int(args.recent_tokens),
                    "recent_decay": float(args.recent_decay),
                    "skip_count": skip_count,
                    "supervision_type": supervision_type,
                    "set_loss_type": args.set_loss_type if supervision_type == "skip_set" else None,
                    "label_search": label_metadata.get("search"),
                    "protected_head": label_metadata.get("protected_head"),
                    "protected_tail": label_metadata.get("protected_tail"),
                    "allowed_layers": allowed_layers,
                    "risk_label_rows": len(risk_labels),
                    "covered_rows": len(covered_indices),
                    "device": str(device),
                },
                indent=2,
            )
        )
        os.makedirs(args.output_dir, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0.0
        total_regression = 0.0
        total_ranking = 0.0
        total_skip_set = 0.0
        used_steps = 0
        iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}") if accelerator.is_main_process else dataloader
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_indices = [int(x) for x in batch["index"].detach().cpu().tolist()]
            router_input_ids, router_attention_mask = prompt_only_inputs(
                input_ids,
                attention_mask,
                labels,
                tokenizer.pad_token_id,
            )
            if args.router_input == "prefix_hk":
                state = teacher_prefix_state(
                    model,
                    router_input_ids,
                    router_attention_mask,
                    num_layers,
                    args.prefix_depth,
                    args.risk_pooling,
                )
                pred = router(state)
            elif args.router_input == "prefix_hk_raw_fusion":
                state = fusion_state(
                    model,
                    router_input_ids,
                    router_attention_mask,
                    num_layers,
                    args.prefix_depth,
                    args.recent_tokens,
                    args.recent_decay,
                )
                pred = router(state)
            elif args.router_input == "prefix_hk_raw_last":
                state = last_fusion_state(
                    model,
                    router_input_ids,
                    router_attention_mask,
                    num_layers,
                    args.prefix_depth,
                )
                pred = router(state)
            elif args.router_input in ATTENTION_ROUTER_INPUTS:
                raw_hidden = raw_embedding_hidden_state(model, router_input_ids)
                hk_hidden = teacher_prefix_hidden_state(
                    model,
                    router_input_ids,
                    router_attention_mask,
                    num_layers,
                    args.prefix_depth,
                )
                pred = router(raw_hidden, hk_hidden, router_attention_mask)
            else:
                state = raw_embedding_state(model, router_input_ids, router_attention_mask)
                pred = router(state)
            target = torch.tensor([risk_labels[index] for index in batch_indices], dtype=torch.float32, device=device)
            if supervision_type == "skip_set":
                regression = pred.new_tensor(0.0)
                ranking = pred.new_tensor(0.0)
                if args.set_loss_type == "exact_k_ce":
                    skip_set = exact_k_subset_ce_loss(
                        pred,
                        target,
                        allowed_layers=allowed_layers,
                        skip_count=int(label_metadata.get("skip_count") or skip_count),
                    )
                else:
                    skip_set = F.binary_cross_entropy_with_logits(-pred.float(), target.float())
                loss = skip_set
            else:
                regression = risk_regression_loss(pred, target, beta=args.huber_beta)
                ranking = risk_pairwise_ranking_loss(pred, target) if args.ranking_loss_weight > 0 else pred.new_tensor(0.0)
                skip_set = risk_skip_set_loss(pred, target, skip_count) if args.skip_set_loss_weight > 0 else pred.new_tensor(0.0)
                loss = (
                    regression
                    + float(args.ranking_loss_weight) * ranking
                    + float(args.skip_set_loss_weight) * skip_set
                )

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss += float(loss.detach().float().item())
            total_regression += float(regression.detach().float().item())
            total_ranking += float(ranking.detach().float().item())
            total_skip_set += float(skip_set.detach().float().item())
            used_steps += 1
            if accelerator.is_main_process:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                if args.loss_log_interval > 0 and used_steps % args.loss_log_interval == 0:
                    denom = max(1, used_steps)
                    print(
                        f"LOSS DEBUG epoch={epoch + 1} step={used_steps} "
                        f"loss={total_loss / denom:.6f} "
                        f"huber={total_regression / denom:.6f} "
                        f"ranking={total_ranking / denom:.6f} "
                        f"skip_set={total_skip_set / denom:.6f}"
                    )

        denom = max(1, used_steps)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / denom,
            "huber": total_regression / denom,
            "ranking": total_ranking / denom,
            "skip_set": total_skip_set / denom,
            "steps": used_steps,
        }
        history.append(epoch_metrics)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} finished: {json.dumps(epoch_metrics)}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(router)
        checkpoint_path = Path(args.output_dir) / "risk_router.pt"
        metadata = {
            "method": (
                "final_kl_greedy_set_supervision"
                if supervision_type == "skip_set"
                else "prefix_supervised_layer_risk_prediction"
            ),
            "router_input": args.router_input,
            "prompt_only_router_context": True,
            "prefix_depth": router_prefix_depth(args.router_input, args.prefix_depth),
            "risk_pooling": router_pooling_metadata(args.router_input, args.risk_pooling),
            "router_features": router_features_metadata(args.router_input),
            "dataset_prompt": "EvalSidDataset",
            "num_layers": int(num_layers),
            "hidden_size": int(state_size),
            "base_hidden_size": int(hidden_size),
            "state_size": int(state_size),
            "router_architecture": router_architecture,
            "router_dim": int(args.router_dim) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "router_heads": int(args.router_heads) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "use_hk_last_residual": args.router_input in {
                "prefix_hk_raw_attn_hk_last_resid",
                "prefix_hk_raw_attn_raw_hk_last_resid",
            },
            "use_raw_last_residual": args.router_input == "prefix_hk_raw_attn_raw_hk_last_resid",
            "recent_tokens": int(args.recent_tokens),
            "recent_decay": float(args.recent_decay),
            "teacher_model": args.teacher_model,
            "risk_label_file": args.risk_label_file,
            "risk_objective": label_metadata.get("objective"),
            "supervision_type": supervision_type,
            "label_search": label_metadata.get("search"),
            "protected_head": label_metadata.get("protected_head"),
            "protected_tail": label_metadata.get("protected_tail"),
            "allowed_layers": allowed_layers,
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "skip_set_loss_weight": float(args.skip_set_loss_weight),
            "set_loss_type": args.set_loss_type if supervision_type == "skip_set" else None,
            "skip_rate": float(args.skip_rate),
            "top_k_layers": int(args.top_k_layers),
            "skip_count": int(skip_count),
            "huber_beta": float(args.huber_beta),
            "seed": int(args.seed),
        }
        torch.save({"model_state_dict": unwrapped.state_dict(), "metadata": metadata}, checkpoint_path)
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        metrics_path.write_text(json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8")
        print(f"Saved risk router checkpoint to {checkpoint_path}")
        print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
