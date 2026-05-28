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
from models.opal_risk_router import PromptCandidateMaskRouter


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


def masked_mean_state(hidden_state, attention_mask):
    weights = attention_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
    return (hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def raw_embedding_state(model, input_ids, attention_mask):
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
    return masked_mean_state(embeds, attention_mask).float()


def load_candidate_labels(path: str):
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[int(row["sample_id"])] = [float(v) for v in row["candidate_losses"]]
    if not labels:
        raise ValueError(f"No candidate labels found in {path}")
    return labels


def load_candidate_metadata(path: str):
    target = Path(path)
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    fallback = target.with_suffix(target.suffix + ".metadata.json")
    if fallback.exists():
        return json.loads(fallback.read_text(encoding="utf-8"))
    raise ValueError(f"Candidate metadata not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train PuDDing-style prompt-only candidate-mask router.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--candidate_label_file", required=True)
    parser.add_argument("--candidate_metadata_file", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    candidate_labels = load_candidate_labels(args.candidate_label_file)
    metadata_path = args.candidate_metadata_file or (args.candidate_label_file + ".metadata.json")
    candidate_metadata = load_candidate_metadata(metadata_path)
    candidate_masks = list(candidate_metadata.get("candidate_masks") or [])
    if not candidate_masks:
        raise ValueError("Candidate metadata must contain candidate_masks.")
    num_candidates = len(candidate_masks)

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
    covered_indices = sorted(index for index in candidate_labels if 0 <= index < len(dataset))
    if not covered_indices:
        raise ValueError("No candidate-label sample_id values match the training dataset indices.")
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
    hidden_size = int(model.config.hidden_size)
    router = PromptCandidateMaskRouter(
        hidden_size=hidden_size,
        num_candidates=num_candidates,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(
            json.dumps(
                {
                    "baseline": "pudding_prompt_candidate",
                    "router_input": "raw_prompt_embedding_mean",
                    "dataset_prompt": "EvalSidDataset",
                    "num_candidates": num_candidates,
                    "candidate_label_rows": len(candidate_labels),
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
            state = raw_embedding_state(model, router_input_ids, router_attention_mask)
            pred = router(state)
            target = torch.tensor([candidate_labels[index] for index in batch_indices], dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(pred.float(), target.float(), beta=float(args.huber_beta))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss += float(loss.detach().float().item())
            used_steps += 1
            if accelerator.is_main_process:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                if args.loss_log_interval > 0 and used_steps % args.loss_log_interval == 0:
                    print(f"LOSS DEBUG epoch={epoch + 1} step={used_steps} huber={total_loss / used_steps:.6f}")

        epoch_metrics = {"epoch": epoch + 1, "loss": total_loss / max(1, used_steps), "steps": used_steps}
        history.append(epoch_metrics)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} finished: {json.dumps(epoch_metrics)}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(router)
        checkpoint_path = Path(args.output_dir) / "candidate_router.pt"
        checkpoint_metadata = {
            "baseline": "pudding_prompt_candidate",
            "method": "prompt_candidate_mask_loss_regression",
            "router_input": "raw_prompt_embedding_mean",
            "prompt_only_router_context": True,
            "dataset_prompt": "EvalSidDataset",
            "teacher_model": args.teacher_model,
            "candidate_label_file": args.candidate_label_file,
            "candidate_metadata_file": metadata_path,
            "candidate_objective": candidate_metadata.get("objective"),
            "candidate_masks": candidate_masks,
            "num_candidates": int(num_candidates),
            "hidden_size": int(hidden_size),
            "skip_rate": candidate_metadata.get("skip_rate"),
            "top_k_layers": candidate_metadata.get("top_k_layers"),
            "num_layers": candidate_metadata.get("num_layers"),
            "huber_beta": float(args.huber_beta),
            "seed": int(args.seed),
        }
        torch.save({"model_state_dict": unwrapped.state_dict(), "metadata": checkpoint_metadata}, checkpoint_path)
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        metrics_path.write_text(json.dumps({"metadata": checkpoint_metadata, "history": history}, indent=2), encoding="utf-8")
        print(f"Saved candidate router checkpoint to {checkpoint_path}")
        print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
