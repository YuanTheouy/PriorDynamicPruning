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
from opal_llm.mask_library import mask_id_from_mask
from opal_llm.related_baselines import deterministic_sample_indices, keep_count_from_args, load_risk_label_rows


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


def simple_kmeans(x: torch.Tensor, k: int, iters: int, seed: int):
    if x.dim() != 2:
        raise ValueError("simple_kmeans expects a 2D tensor")
    n = int(x.size(0))
    if n <= 0:
        raise ValueError("No embeddings for k-means")
    k = max(1, min(int(k), n))
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    init = torch.randperm(n, generator=generator, device=x.device)[:k]
    centers = x.index_select(0, init).clone()
    assignments = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(max(1, int(iters))):
        distances = torch.cdist(x.float(), centers.float(), p=2)
        assignments = distances.argmin(dim=1)
        next_centers = centers.clone()
        for cluster_id in range(k):
            members = x[assignments == cluster_id]
            if members.numel() > 0:
                next_centers[cluster_id] = members.mean(dim=0)
        if torch.allclose(next_centers, centers, atol=1e-5, rtol=1e-5):
            centers = next_centers
            break
        centers = next_centers
    return centers, assignments


def mask_from_average_risk(risks, keep_count: int):
    num_layers = len(risks)
    skip_count = max(0, int(num_layers) - int(keep_count))
    skip_layers = {idx for idx, _ in sorted(enumerate(risks), key=lambda item: (float(item[1]), item[0]))[:skip_count]}
    return [0 if idx in skip_layers else 1 for idx in range(num_layers)]


def main():
    parser = argparse.ArgumentParser(description="Build IG-Pruning-style prompt-cluster mask artifact.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--risk_label_file", required=True)
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--kmeans_iters", type=int, default=50)
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--skip_rate", type=float, default=0.25)
    parser.add_argument("--top_k_layers", type=int, default=21)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    risk_rows = load_risk_label_rows(args.risk_label_file)
    if not risk_rows:
        raise ValueError(f"No risk label rows found in {args.risk_label_file}")

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
            train_file=args.train_file,
            tokenizer=tokenizer,
            category=dataset_category,
            max_len=2560,
            test=False,
            seed=args.seed,
        )
    )
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    selected_indices = deterministic_sample_indices(
        total=len(base_dataset),
        max_samples=args.max_samples,
        seed=sample_seed,
        strategy=args.sample_strategy,
        allowed=risk_rows.keys(),
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

    output_path = Path(args.output)
    if accelerator.is_main_process:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for stale in output_path.parent.glob(output_path.name + ".rank*"):
            stale.unlink()
    accelerator.wait_for_everyone()

    local_embeddings = []
    local_sample_ids = []
    iterator = tqdm(dataloader, desc="ig-embeddings") if accelerator.is_main_process else dataloader
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_ids, prompt_mask = prompt_only_inputs(input_ids, attention_mask, labels, tokenizer.pad_token_id)
            embeddings = raw_embedding_state(model, prompt_ids, prompt_mask).detach().float().cpu()
            local_embeddings.append(embeddings)
            local_sample_ids.extend(int(x) for x in batch["index"].detach().cpu().tolist())

    local_tensor = torch.cat(local_embeddings, dim=0) if local_embeddings else torch.empty(0, int(model.config.hidden_size))
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")
    torch.save({"sample_ids": local_sample_ids, "embeddings": local_tensor}, shard_path)
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    pairs = []
    for path in sorted(output_path.parent.glob(output_path.name + ".rank*")):
        shard = torch.load(path, map_location="cpu")
        pairs.extend(zip(shard.get("sample_ids", []), shard.get("embeddings", [])))
    pairs = sorted(pairs, key=lambda item: int(item[0]))
    sample_ids = [int(sample_id) for sample_id, _ in pairs]
    embeddings = torch.stack([embedding for _, embedding in pairs], dim=0).float()
    if args.normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    first_risk = next(iter(risk_rows.values()))["risk_labels"]
    num_layers = len(first_risk)
    keep_count = keep_count_from_args(num_layers, args.skip_rate, args.top_k_layers)
    centers, assignments = simple_kmeans(embeddings, args.num_clusters, args.kmeans_iters, args.seed)
    if args.normalize_embeddings:
        centers = torch.nn.functional.normalize(centers, p=2, dim=-1)

    global_risk = []
    for layer_idx in range(num_layers):
        values = [float(row["risk_labels"][layer_idx]) for row in risk_rows.values() if len(row.get("risk_labels", [])) == num_layers]
        global_risk.append(sum(values) / max(1, len(values)))

    masks = []
    mask_ids = []
    cluster_sizes = []
    cluster_risks = []
    for cluster_id in range(int(centers.size(0))):
        member_ids = [sample_ids[row_idx] for row_idx, value in enumerate(assignments.tolist()) if int(value) == cluster_id]
        cluster_sizes.append(len(member_ids))
        avg_risk = []
        for layer_idx in range(num_layers):
            values = [
                float(risk_rows[sample_id]["risk_labels"][layer_idx])
                for sample_id in member_ids
                if sample_id in risk_rows and len(risk_rows[sample_id].get("risk_labels", [])) == num_layers
            ]
            avg_risk.append(sum(values) / len(values) if values else global_risk[layer_idx])
        mask = mask_from_average_risk(avg_risk, keep_count=keep_count)
        masks.append(mask)
        mask_ids.append(mask_id_from_mask(mask, prefix=f"ig_cluster_c{cluster_id}"))
        cluster_risks.append(avg_risk)

    metadata = {
        "baseline": "ig_cluster_mask",
        "method": "prompt_cluster_average_risk_mask",
        "router_input": "raw_prompt_embedding_mean",
        "prompt_only_router_context": True,
        "teacher_model": args.teacher_model,
        "train_file": args.train_file,
        "risk_label_file": args.risk_label_file,
        "num_clusters": int(centers.size(0)),
        "kmeans_iters": int(args.kmeans_iters),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "num_layers": int(num_layers),
        "top_k_layers": int(keep_count),
        "skip_rate": float(args.skip_rate),
        "cluster_sizes": cluster_sizes,
        "mask_ids": mask_ids,
        "sample_ids": sample_ids,
        "sample_strategy": args.sample_strategy,
        "sample_seed": int(sample_seed),
        "seed": int(args.seed),
    }
    artifact = {
        "centers": centers.cpu(),
        "masks": torch.tensor(masks, dtype=torch.long),
        "cluster_risks": torch.tensor(cluster_risks, dtype=torch.float32),
        "metadata": metadata,
    }
    torch.save(artifact, output_path)
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved IG cluster artifact to {output_path}")


if __name__ == "__main__":
    main()
