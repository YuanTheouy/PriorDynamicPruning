#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

from transformers import AutoTokenizer, Qwen2ForCausalLM

from data import EvalSidDataset
from models.opal_risk_router import LayerwiseHiddenRiskRouter, risk_pairwise_ranking_loss, risk_regression_loss, risk_skip_set_loss
from opal_llm.mask_utils import keep_count_from_skip_rate


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


def prompt_layerwise_states(model, input_ids, attention_mask, num_layers: int, pooling: str):
    clear_custom_policy(model)
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < num_layers:
        raise ValueError("Model did not return enough hidden states.")
    # hidden_states[l] is the prompt-only state available before deciding layer l.
    pooled = [
        pool_hidden_state(hidden_states[layer_idx], attention_mask, pooling=pooling)
        for layer_idx in range(num_layers)
    ]
    return torch.stack(pooled, dim=1).float()


def load_risk_labels(path: str):
    labels = {}
    metadata = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[int(row["sample_id"])] = [float(v) for v in row["risk_labels"]]
            metadata.setdefault("objective", row.get("objective"))
            metadata.setdefault("num_layers", row.get("num_layers"))
    if not labels:
        raise ValueError(f"No risk labels found in {path}")
    return labels, metadata


def main():
    parser = argparse.ArgumentParser(description="Train Dr.LLM-style layer-wise hidden-state risk router.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--risk_label_file", required=True)
    parser.add_argument("--risk_pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--skip_set_loss_weight", type=float, default=0.0)
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

    risk_labels, label_metadata = load_risk_labels(args.risk_label_file)
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
    keep_count = keep_count_from_skip_rate(
        num_layers,
        args.skip_rate,
        args.top_k_layers if args.top_k_layers > 0 else num_layers,
    )
    skip_count = num_layers - keep_count
    router = LayerwiseHiddenRiskRouter(hidden_size=hidden_size, num_layers=num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(
            json.dumps(
                {
                    "baseline": "layerwise_hidden_router",
                    "router_input": "prompt_layerwise_hidden_states",
                    "risk_pooling": args.risk_pooling,
                    "dataset_prompt": "EvalSidDataset",
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "skip_count": skip_count,
                    "risk_label_rows": len(risk_labels),
                    "covered_rows": len(covered_indices),
                    "device": str(device),
                },
                indent=2,
            )
        )

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
            states = prompt_layerwise_states(
                model,
                router_input_ids,
                router_attention_mask,
                num_layers=num_layers,
                pooling=args.risk_pooling,
            )
            pred = router(states)
            target = torch.tensor([risk_labels[index] for index in batch_indices], dtype=torch.float32, device=device)
            regression = risk_regression_loss(pred, target, beta=args.huber_beta)
            ranking = risk_pairwise_ranking_loss(pred, target) if args.ranking_loss_weight > 0 else pred.new_tensor(0.0)
            skip_set = risk_skip_set_loss(pred, target, skip_count) if args.skip_set_loss_weight > 0 else pred.new_tensor(0.0)
            loss = regression + float(args.ranking_loss_weight) * ranking + float(args.skip_set_loss_weight) * skip_set

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
        checkpoint_path = Path(args.output_dir) / "layerwise_hidden_router.pt"
        metadata = {
            "baseline": "layerwise_hidden_router",
            "method": "layerwise_hidden_state_risk_prediction",
            "router_input": "prompt_layerwise_hidden_states",
            "prompt_only_router_context": True,
            "risk_pooling": args.risk_pooling,
            "dataset_prompt": "EvalSidDataset",
            "num_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "teacher_model": args.teacher_model,
            "risk_label_file": args.risk_label_file,
            "risk_objective": label_metadata.get("objective"),
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "skip_set_loss_weight": float(args.skip_set_loss_weight),
            "skip_rate": float(args.skip_rate),
            "top_k_layers": int(args.top_k_layers),
            "skip_count": int(skip_count),
            "huber_beta": float(args.huber_beta),
            "seed": int(args.seed),
        }
        torch.save({"model_state_dict": unwrapped.state_dict(), "metadata": metadata}, checkpoint_path)
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        metrics_path.write_text(json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8")
        print(f"Saved layerwise hidden router checkpoint to {checkpoint_path}")
        print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
