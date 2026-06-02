#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)
sys.path.insert(0, current_dir)

from transformers import AutoTokenizer, Qwen2ForCausalLM

from models.opal_risk_router import LayerQueryCrossAttentionRiskRouter, OpalRiskRouter
from wikitext_opal_utils import (
    C6_STATIC_STRATEGIES,
    aggregate_loss_rows,
    allowed_layers_from_policy,
    collate_wikitext_windows,
    detect_num_layers,
    dtype_from_precision,
    finite_exp,
    keep_masks_from_skip_risk,
    lm_loss_stats_from_logits,
    load_wikitext_token_ids,
    mask_key,
    read_jsonl,
    resolve_skip_budget,
    skipped_layers_from_keep_mask,
    static_keep_mask,
    summarize_keep_masks,
    WikitextWindowDataset,
    write_json,
)


ATTENTION_ROUTER_INPUTS = {"prefix_hk_raw_attn"}


def router_prefix_batch(input_ids, attention_mask, router_prefix_tokens: int):
    prefix = max(1, min(int(router_prefix_tokens), input_ids.size(1) - 1))
    return input_ids[:, :prefix].contiguous(), attention_mask[:, :prefix].contiguous()


def masked_mean_state(hidden_state, attention_mask):
    weights = attention_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
    return (hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def raw_embedding_hidden_state(model, input_ids):
    with torch.no_grad():
        return model.model.embed_tokens(input_ids)


def raw_embedding_state(model, input_ids, attention_mask):
    return masked_mean_state(raw_embedding_hidden_state(model, input_ids), attention_mask).float()


def teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int):
    prefix_depth = max(0, min(int(prefix_depth), int(num_layers)))
    prefix_mask = torch.tensor(
        [[1 if idx < prefix_depth else 0 for idx in range(int(num_layers))]] * input_ids.size(0),
        dtype=torch.float32,
        device=input_ids.device,
    )
    with torch.no_grad():
        return model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_mask=prefix_mask,
            use_cache=False,
        ).last_hidden_state


def build_router(
    router_input: str,
    hidden_size: int,
    num_layers: int,
    router_dim: int,
    router_heads: int,
    dropout: float = 0.0,
):
    if router_input == "prefix_hk_raw_attn":
        return LayerQueryCrossAttentionRiskRouter(
            base_hidden_size=hidden_size,
            num_layers=num_layers,
            router_dim=router_dim,
            router_heads=router_heads,
            dropout=dropout,
            budget_condition="none",
            max_budget=num_layers,
        )
    if router_input == "raw_embedding":
        return OpalRiskRouter(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unsupported router_input: {router_input}")


def router_forward(
    router,
    router_input: str,
    model,
    input_ids,
    attention_mask,
    num_layers: int,
    prefix_depth: int,
):
    if router_input == "raw_embedding":
        return router(raw_embedding_state(model, input_ids, attention_mask))
    if router_input == "prefix_hk_raw_attn":
        raw_hidden = raw_embedding_hidden_state(model, input_ids)
        hk_hidden = teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers, prefix_depth)
        return router(raw_hidden, hk_hidden, attention_mask)
    raise ValueError(f"Unsupported router_input: {router_input}")


def load_router_checkpoint(path: str, hidden_size: int, num_layers: int, device):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    metadata = dict(checkpoint.get("metadata") or {})
    router_input = str(metadata.get("router_input") or "raw_embedding")
    router = build_router(
        router_input=router_input,
        hidden_size=int(metadata.get("base_hidden_size") or hidden_size),
        num_layers=num_layers,
        router_dim=int(metadata.get("router_dim") or 256),
        router_heads=int(metadata.get("router_heads") or 4),
        dropout=0.0,
    )
    router.load_state_dict(state)
    router.to(device).eval()
    return router, metadata


def forward_loss_rows(model, input_ids, attention_mask, labels, layer_mask=None):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer_mask=layer_mask,
        use_cache=False,
    )
    rows = lm_loss_stats_from_logits(outputs.logits, labels)
    del outputs
    return rows


def load_label_rows(path: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: int(row["sample_id"]))
    if not rows:
        raise ValueError(f"No label rows found in {path}")
    metadata_path = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in [
        "num_layers",
        "skip_rate",
        "skip_count",
        "keep_count",
        "protected_head",
        "protected_tail",
        "allowed_layers",
        "seq_len",
        "router_prefix_tokens",
        "objective",
        "search",
    ]:
        if key not in metadata and rows[0].get(key) is not None:
            metadata[key] = rows[0].get(key)
    return rows, metadata


def make_dataset_for_split(args, tokenizer, selected_window_ids=None, max_windows=0):
    token_ids = load_wikitext_token_ids(
        tokenizer,
        split=args.split,
        dataset_disk_path=getattr(args, "dataset_disk_path", ""),
        dataset_cache_dir=getattr(args, "dataset_cache_dir", ""),
        dataset_path=getattr(args, "dataset_path", "wikitext"),
        dataset_name=getattr(args, "dataset_name", "wikitext-2-raw-v1"),
    )
    dataset = WikitextWindowDataset(
        token_ids,
        seq_len=args.seq_len,
        max_windows=max_windows,
        sample_strategy=getattr(args, "sample_strategy", "first"),
        sample_seed=getattr(args, "sample_seed", args.seed),
        selected_window_ids=selected_window_ids,
    )
    return dataset, token_ids


def train_router(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    label_rows, label_metadata = load_label_rows(args.risk_label_file)
    label_by_sample = {int(row["sample_id"]): row for row in label_rows}
    selected_window_ids = sorted(label_by_sample)
    dataset, token_ids = make_dataset_for_split(
        args,
        tokenizer,
        selected_window_ids=selected_window_ids,
        max_windows=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_wikitext_windows(batch, args.router_prefix_tokens),
    )

    model = Qwen2ForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    skip_count = int(label_metadata.get("skip_count") or resolve_skip_budget(num_layers, args.skip_rate, args.skip_count)[0])
    keep_count = int(num_layers) - skip_count
    allowed_layers = label_metadata.get("allowed_layers") or allowed_layers_from_policy(
        num_layers,
        args.protected_head,
        args.protected_tail,
    )
    router = build_router(
        router_input=args.router_input,
        hidden_size=hidden_size,
        num_layers=num_layers,
        router_dim=args.router_dim,
        router_heads=args.router_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "task": "wikitext2_public_lm_sanity_train_router",
                    "router_input": args.router_input,
                    "teacher_model": args.teacher_model,
                    "dataset_disk_path": args.dataset_disk_path,
                    "num_layers": int(num_layers),
                    "hidden_size": int(hidden_size),
                    "skip_count": int(skip_count),
                    "keep_count": int(keep_count),
                    "protected_head": int(args.protected_head),
                    "protected_tail": int(args.protected_tail),
                    "allowed_layers": [int(idx) for idx in allowed_layers],
                    "label_rows": len(label_rows),
                    "covered_rows": len(selected_window_ids),
                    "seq_len": int(args.seq_len),
                    "router_prefix_tokens": int(args.router_prefix_tokens),
                    "token_count_train_split": int(len(token_ids)),
                    "target_leakage_guard": "router sees only first router_prefix_tokens; NLL labels score suffix tokens only",
                },
                indent=2,
            )
        )

    history = []
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0.0
        total_steps = 0
        iterator = tqdm(dataloader, desc=f"{args.router_input} epoch {epoch + 1}/{args.epochs}") if accelerator.is_main_process else dataloader
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            router_input_ids, router_attention_mask = router_prefix_batch(
                input_ids,
                attention_mask,
                args.router_prefix_tokens,
            )
            pred = router_forward(
                router,
                args.router_input,
                model,
                router_input_ids,
                router_attention_mask,
                num_layers,
                args.prefix_depth,
            )
            target = torch.tensor(
                [label_by_sample[sample_id]["skip_mask"] for sample_id in sample_ids],
                dtype=torch.float32,
                device=device,
            )
            loss = F.binary_cross_entropy_with_logits(-pred.float(), target.float())
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite router loss: {float(loss.detach().float().item())}")
            optimizer.zero_grad()
            accelerator.backward(loss)
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(router.parameters(), float(args.max_grad_norm))
            optimizer.step()
            total_loss += float(loss.detach().float().item())
            total_steps += 1
            if accelerator.is_main_process:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_tensor = torch.tensor([total_loss, float(total_steps)], device=device)
        gathered = accelerator.gather(epoch_tensor.unsqueeze(0))
        summed = gathered.sum(dim=0)
        global_steps = max(1.0, float(summed[1].item()))
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": float(summed[0].item() / global_steps),
            "skip_set": float(summed[0].item() / global_steps),
            "steps": int(global_steps),
        }
        history.append(epoch_metrics)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} finished: {json.dumps(epoch_metrics)}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(router)
        metadata = {
            "method": "wikitext2_delta_nll_greedy_set_bce",
            "router_input": args.router_input,
            "prompt_only_router_context": True,
            "target_leakage_guard": "router sees only first router_prefix_tokens; eval never loads greedy labels",
            "prefix_depth": int(args.prefix_depth) if args.router_input in ATTENTION_ROUTER_INPUTS else 0,
            "router_features": (
                ["raw_token_sequence", "hk_token_sequence", "attention_mask", "layer_queries"]
                if args.router_input == "prefix_hk_raw_attn"
                else ["raw_embedding_mean"]
            ),
            "router_architecture": (
                "layer_query_cross_attention" if args.router_input == "prefix_hk_raw_attn" else "pooled_mlp"
            ),
            "router_dim": int(args.router_dim) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "router_heads": int(args.router_heads) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "teacher_model": args.teacher_model,
            "risk_label_file": args.risk_label_file,
            "risk_objective": label_metadata.get("objective", "Delta_NLL"),
            "supervision_type": "skip_set",
            "set_loss_type": "bce",
            "label_search": label_metadata.get("search", "forward_greedy"),
            "dataset": "wikitext-2-raw-v1",
            "dataset_disk_path": args.dataset_disk_path,
            "split": args.split,
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "num_layers": int(num_layers),
            "base_hidden_size": int(hidden_size),
            "hidden_size": int(hidden_size),
            "state_size": int(hidden_size),
            "skip_rate": float(args.skip_rate),
            "skip_count": int(skip_count),
            "keep_count": int(keep_count),
            "protected_head": int(args.protected_head),
            "protected_tail": int(args.protected_tail),
            "allowed_layers": [int(idx) for idx in allowed_layers],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "seed": int(args.seed),
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": unwrapped.state_dict(), "metadata": metadata}, output_dir / "risk_router.pt")
        (output_dir / "training_metrics.json").write_text(
            json.dumps({"metadata": metadata, "history": history}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved router checkpoint to {output_dir / 'risk_router.pt'}")


def eval_method(args):
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, token_ids = make_dataset_for_split(
        args,
        tokenizer,
        selected_window_ids=None,
        max_windows=args.eval_windows,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_wikitext_windows(batch, args.router_prefix_tokens),
    )
    dataloader = accelerator.prepare(dataloader)

    model = Qwen2ForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    skip_count, keep_count = resolve_skip_budget(num_layers, args.skip_rate, args.skip_count)
    allowed_layers = allowed_layers_from_policy(num_layers, args.protected_head, args.protected_tail)
    router = None
    router_metadata = {}
    if args.method == "router":
        router, router_metadata = load_router_checkpoint(args.risk_router_ckpt, hidden_size, num_layers, device)
        allowed_layers = router_metadata.get("allowed_layers") or allowed_layers
        skip_count = int(router_metadata.get("skip_count") or skip_count)
        keep_count = int(num_layers) - skip_count
    static_mask = None
    if args.method == "static":
        static_mask = static_keep_mask(
            args.static_strategy,
            num_layers=num_layers,
            skip_count=skip_count,
            protected_head=args.protected_head,
            protected_tail=args.protected_tail,
            seed=args.seed,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_path.unlink(missing_ok=True)
        for stale in output_path.parent.glob(output_path.name + ".rank*"):
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")

    iterator = tqdm(dataloader, desc=f"eval-{args.method}-{args.split}") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            window_starts = [int(x) for x in batch["window_start"].detach().cpu().tolist()]

            layer_mask = None
            keep_masks = []
            if args.method == "full":
                keep_masks = [[1] * int(num_layers) for _ in sample_ids]
            elif args.method == "static":
                keep_masks = [list(static_mask) for _ in sample_ids]
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=device)
            elif args.method == "router":
                router_input_ids, router_attention_mask = router_prefix_batch(
                    input_ids,
                    attention_mask,
                    args.router_prefix_tokens,
                )
                pred_risk = router_forward(
                    router,
                    str(router_metadata.get("router_input") or "raw_embedding"),
                    model,
                    router_input_ids,
                    router_attention_mask,
                    num_layers,
                    int(router_metadata.get("prefix_depth") or args.prefix_depth),
                )
                keep_masks = keep_masks_from_skip_risk(pred_risk, skip_count=skip_count, allowed_layers=allowed_layers)
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unsupported eval method: {args.method}")

            rows = forward_loss_rows(model, input_ids, attention_mask, labels, layer_mask=layer_mask)
            for sample_id, window_start, stats, keep_mask in zip(sample_ids, window_starts, rows, keep_masks):
                row = {
                    "sample_id": int(sample_id),
                    "window_start": int(window_start),
                    "total_nll": float(stats["total_nll"]),
                    "token_count": int(stats["token_count"]),
                    "nll": float(stats["nll"]),
                    "ppl": float(stats["ppl"]),
                    "keep_mask": [int(v) for v in keep_mask],
                    "mask_key": mask_key(keep_mask),
                    "skipped_layers": skipped_layers_from_keep_mask(keep_mask),
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
        loss_summary = aggregate_loss_rows(rows)
        expected_skip_count = 0 if args.method == "full" else skip_count
        mask_summary = summarize_keep_masks([row["keep_mask"] for row in rows], expected_skip_count=expected_skip_count)
        if not math.isfinite(float(loss_summary["ppl"])):
            raise FloatingPointError(f"Non-finite PPL for {args.run_name}: {loss_summary['ppl']}")
        if args.method in {"static", "router"} and float(mask_summary["exact_skip_count_rate"]) < 1.0:
            raise ValueError(
                f"{args.run_name} did not produce exactly K skipped layers for every window: "
                f"exact_skip_count_rate={mask_summary['exact_skip_count_rate']}"
            )
        payload = {
            "run_name": args.run_name,
            "method": args.method,
            "method_label": args.method_label,
            "static_strategy": args.static_strategy if args.method == "static" else None,
            "risk_router_ckpt": args.risk_router_ckpt if args.method == "router" else "",
            "risk_router_metadata": router_metadata if args.method == "router" else {},
            "teacher_model": args.teacher_model,
            "model_name": os.path.basename(os.path.normpath(args.teacher_model)),
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "dataset_disk_path": args.dataset_disk_path,
            "split": args.split,
            "num_tokens_in_split": int(len(token_ids)),
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "eval_windows_requested": int(args.eval_windows),
            "eval_windows": int(len(rows)),
            "eval_tokens": int(loss_summary["eval_tokens"]),
            "num_layers": int(num_layers),
            "skip_rate": float(args.skip_rate),
            "skip_count": int(skip_count if args.method != "full" else 0),
            "keep_count": int(num_layers if args.method == "full" else keep_count),
            "protected_head": int(args.protected_head),
            "protected_tail": int(args.protected_tail),
            "allowed_layers": [int(idx) for idx in allowed_layers],
            "nll": float(loss_summary["nll"]),
            "ppl": float(loss_summary["ppl"]),
            "total_nll": float(loss_summary["total_nll"]),
            "mask_summary": mask_summary,
            "unique_masks": int(mask_summary["unique_masks"]),
            "average_kept_layers": float(mask_summary["average_kept_layers"]),
            "average_skipped_layers": float(mask_summary["average_skipped_layers"]),
            "exact_skip_count_rate": float(mask_summary["exact_skip_count_rate"]),
            "uses_greedy_labels_at_eval": False,
            "target_leakage_guard": "router sees only first router_prefix_tokens; PPL scores suffix tokens only",
            "rows": rows if args.save_rows_in_json else [],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ["run_name", "method_label", "split", "nll", "ppl", "eval_tokens", "unique_masks"]}, indent=2))
        print(f"Wrote eval metrics to {output_path}")


def diagnose_overlap(args):
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    label_rows, label_metadata = load_label_rows(args.risk_label_file)
    if args.max_samples > 0 and args.max_samples < len(label_rows):
        if args.sample_strategy == "first":
            label_rows = label_rows[: args.max_samples]
        elif args.sample_strategy == "random":
            import random

            rng = random.Random(args.sample_seed)
            label_rows = sorted(rng.sample(label_rows, args.max_samples), key=lambda row: int(row["sample_id"]))
        else:
            raise ValueError(f"Unsupported sample_strategy: {args.sample_strategy}")
    label_by_sample = {int(row["sample_id"]): row for row in label_rows}
    selected_window_ids = sorted(label_by_sample)
    dataset, _ = make_dataset_for_split(
        args,
        tokenizer,
        selected_window_ids=selected_window_ids,
        max_windows=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_wikitext_windows(batch, args.router_prefix_tokens),
    )
    dataloader = accelerator.prepare(dataloader)

    model = Qwen2ForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    router, router_metadata = load_router_checkpoint(args.risk_router_ckpt, hidden_size, num_layers, device)
    skip_count = int(label_metadata.get("skip_count") or router_metadata.get("skip_count"))
    allowed_layers = label_metadata.get("allowed_layers") or router_metadata.get("allowed_layers")
    if not allowed_layers:
        allowed_layers = allowed_layers_from_policy(num_layers, args.protected_head, args.protected_tail)

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        for stale in output_path.parent.glob(output_path.name + ".rank*"):
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")

    iterator = tqdm(dataloader, desc="wikitext-overlap") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            router_input_ids, router_attention_mask = router_prefix_batch(
                input_ids,
                attention_mask,
                args.router_prefix_tokens,
            )
            pred_risk = router_forward(
                router,
                str(router_metadata.get("router_input") or "raw_embedding"),
                model,
                router_input_ids,
                router_attention_mask,
                num_layers,
                int(router_metadata.get("prefix_depth") or args.prefix_depth),
            )
            pred_masks = keep_masks_from_skip_risk(pred_risk, skip_count=skip_count, allowed_layers=allowed_layers)
            for sample_id, pred_keep_mask in zip(sample_ids, pred_masks):
                label_row = label_by_sample[int(sample_id)]
                label_skip = {idx for idx, value in enumerate(label_row["skip_mask"]) if int(value) == 1}
                pred_skip = set(skipped_layers_from_keep_mask(pred_keep_mask))
                overlap = len(label_skip & pred_skip)
                hamming = sum(
                    1
                    for idx in range(int(num_layers))
                    if (idx in label_skip) != (idx in pred_skip)
                )
                row = {
                    "sample_id": int(sample_id),
                    "router_input": router_metadata.get("router_input"),
                    "num_layers": int(num_layers),
                    "skip_count": int(skip_count),
                    "predicted_skipped_layers": sorted(pred_skip),
                    "label_skipped_layers": sorted(label_skip),
                    "overlap_count": int(overlap),
                    "overlap_ratio": float(overlap / max(1, skip_count)),
                    "hamming_count": int(hamming),
                    "hamming_ratio": float(hamming / max(1, int(num_layers))),
                    "exact_match": int(pred_skip == label_skip),
                    "predicted_skip_count": int(len(pred_skip)),
                    "label_skip_count": int(len(label_skip)),
                    "mask_key": mask_key(pred_keep_mask),
                }
                f.write(json.dumps(row) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        row_by_sample = {}
        for path in sorted(output_path.parent.glob(output_path.name + ".rank*")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        row_by_sample.setdefault(int(row["sample_id"]), row)
        rows = [row_by_sample[key] for key in sorted(row_by_sample)]

        def mean(key):
            return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))

        unique_masks = len({row["mask_key"] for row in rows})
        summary = {
            "num_samples": int(len(rows)),
            "router_input": router_metadata.get("router_input"),
            "risk_router_ckpt": args.risk_router_ckpt,
            "risk_label_file": args.risk_label_file,
            "num_layers": int(num_layers),
            "skip_count": int(skip_count),
            "mean_overlap_count": mean("overlap_count"),
            "mean_overlap_ratio": mean("overlap_ratio"),
            "mean_hamming_count": mean("hamming_count"),
            "mean_hamming_ratio": mean("hamming_ratio"),
            "exact_match_rate": mean("exact_match"),
            "mean_predicted_skip_count": mean("predicted_skip_count"),
            "mean_label_skip_count": mean("label_skip_count"),
            "unique_predicted_masks": int(unique_masks),
            "allowed_layers": [int(idx) for idx in allowed_layers],
            "target_leakage_guard": "diagnostic uses train labels only; eval/test path never loads labels",
        }
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Wrote overlap summary to {summary_path}")


def parse_named_paths(values: Sequence[str]) -> Dict[str, str]:
    result = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def load_optional_json(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def training_loss_summary(path: str):
    payload = load_optional_json(path)
    if not payload:
        return None
    history = payload.get("history") or []
    if not history:
        return None
    first = history[0]
    last = history[-1]
    best = min(history, key=lambda row: float(row.get("loss", float("inf"))))
    return {
        "first": first,
        "best": best,
        "last": last,
        "epochs": len(history),
        "metadata": payload.get("metadata", {}),
    }


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(args):
    metric_paths = parse_named_paths(args.metric)
    metrics = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in metric_paths.items()}
    if "Full" not in metrics:
        raise ValueError("write_report requires --metric Full=/path/full.json")
    full = metrics["Full"]
    full_nll = float(full["nll"])
    full_ppl = float(full["ppl"])
    label_metadata = load_optional_json(args.label_metadata) or {}
    raw_train = training_loss_summary(args.raw_training_metrics)
    opal_train = training_loss_summary(args.opal_training_metrics)
    raw_overlap = load_optional_json(args.raw_overlap_summary)
    opal_overlap = load_optional_json(args.opal_overlap_summary)

    order = [
        "Full",
        "Static uniform",
        "Static ends_heavy",
        "Static best-on-val C6",
        "Raw-SetBCE",
        "OPAL-SetBCE",
    ]
    lines = []
    lines.append("# WikiText-2 Public LM Sanity Results")
    lines.append("")
    lines.append(f"Last updated: {args.date}")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Server run completed and this file was generated by `eval_wikitext_opal_ppl.py write_report`.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- model path: `{full.get('teacher_model')}`")
    lines.append(f"- model name: `{full.get('model_name')}`")
    lines.append(f"- num_layers: {full.get('num_layers')}")
    dataset_source = full.get("dataset_disk_path") or f"{full.get('dataset_path')}/{full.get('dataset_name')}"
    lines.append(f"- dataset: `{dataset_source}`")
    lines.append(f"- eval split: `test`; static selection split: `validation`; label split: `train`")
    lines.append(f"- seq_len: {full.get('seq_len')}")
    lines.append(f"- router_prefix_tokens: {full.get('router_prefix_tokens')}")
    lines.append(f"- eval_windows: {full.get('eval_windows')}")
    lines.append(f"- eval_tokens: {full.get('eval_tokens')}")
    lines.append(f"- skip_rate: {label_metadata.get('skip_rate', full.get('skip_rate'))}")
    lines.append(f"- K skipped: {label_metadata.get('skip_count', full.get('skip_count'))}")
    lines.append(f"- kept_layers: {label_metadata.get('keep_count', full.get('keep_count'))}")
    lines.append(f"- protected_head / protected_tail: {full.get('protected_head')} / {full.get('protected_tail')}")
    lines.append(f"- allowed_layers: `{label_metadata.get('allowed_layers', full.get('allowed_layers'))}`")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| method | K skipped | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in order:
        if name not in metrics:
            continue
        row = metrics[name]
        nll = float(row["nll"])
        ppl = float(row["ppl"])
        delta_nll = nll - full_nll
        delta_ppl = ppl - full_ppl
        k_skip = 0 if name == "Full" else int(row.get("skip_count", 0))
        unique = row.get("unique_masks", 1)
        lines.append(
            f"| {name} | {k_skip} | {fmt(nll)} | {fmt(ppl)} | {fmt(delta_nll)} | {fmt(delta_ppl)} | {unique} |"
        )
    lines.append("")
    lines.append("## Static Selection")
    lines.append("")
    best_metric = metrics.get("Static best-on-val C6", {})
    lines.append(f"- selected C6 strategy on validation: `{args.static_best_strategy or best_metric.get('static_strategy')}`")
    if args.static_best_val_json:
        best_val = load_optional_json(args.static_best_val_json) or {}
        if best_val:
            lines.append(f"- validation NLL/PPL for selected static: {fmt(best_val.get('nll'))} / {fmt(best_val.get('ppl'))}")
    lines.append("- C6 candidates: `uniform`, `ends_heavy`, `first_k`, `last_k`, `middle_heavy`, `random_diverse_seed42`")
    lines.append("- Test split is not used to choose the static mask.")
    lines.append("")
    lines.append("## Router Training")
    lines.append("")
    for label, summary in [("Raw-SetBCE", raw_train), ("OPAL-SetBCE", opal_train)]:
        if not summary:
            continue
        first = summary["first"]
        best = summary["best"]
        last = summary["last"]
        lines.append(
            f"- {label}: epochs={summary['epochs']}, "
            f"first loss={fmt(first.get('loss'))}, "
            f"best loss={fmt(best.get('loss'))} @ epoch {best.get('epoch')}, "
            f"last loss={fmt(last.get('loss'))}"
        )
    lines.append("")
    lines.append("## Train-Label Overlap")
    lines.append("")
    for label, summary in [("Raw-SetBCE", raw_overlap), ("OPAL-SetBCE", opal_overlap)]:
        if not summary:
            continue
        lines.append(
            f"- {label}: overlap@K={fmt(summary.get('mean_overlap_ratio'))}, "
            f"hamming={fmt(summary.get('mean_hamming_ratio'))}, "
            f"unique masks={summary.get('unique_predicted_masks')}"
        )
    lines.append("")
    lines.append("## Leakage Check")
    lines.append("")
    lines.append("- Greedy labels are built only from WikiText-2 train windows.")
    lines.append("- Static best-on-val C6 is selected only on validation PPL.")
    lines.append("- Final PPL is reported on test windows.")
    lines.append("- Eval code does not load greedy labels for Full/static/router PPL.")
    lines.append("- Router masks are predicted from the first `router_prefix_tokens` of each window; PPL is scored only on the suffix tokens.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if "OPAL-SetBCE" in metrics:
        opal_delta = float(metrics["OPAL-SetBCE"]["nll"]) - full_nll
        static_names = [name for name in ["Static uniform", "Static ends_heavy", "Static best-on-val C6"] if name in metrics]
        wins_static = all(opal_delta < float(metrics[name]["nll"]) - full_nll for name in static_names)
        if wins_static:
            lines.append("OPAL-SetBCE improves over the fixed static skipping baselines on WikiText-2 PPL.")
        else:
            lines.append("OPAL-SetBCE did not beat every fixed static baseline; treat WikiText-2 as a diagnostic sanity result and inspect implementation/model mismatch before using it as a main claim.")
        if "Raw-SetBCE" in metrics:
            raw_delta = float(metrics["Raw-SetBCE"]["nll"]) - full_nll
            if opal_delta < raw_delta:
                lines.append("OPAL-SetBCE also improves over Raw-SetBCE in this run.")
            else:
                lines.append("OPAL-SetBCE does not improve over Raw-SetBCE in this run; the prefix-vs-raw gap should be described as task dependent.")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate WikiText-2 OPAL PPL sanity benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p):
        p.add_argument("--teacher_model", required=True)
        p.add_argument("--split", default="train")
        p.add_argument("--dataset_path", default="wikitext")
        p.add_argument("--dataset_name", default="wikitext-2-raw-v1")
        p.add_argument("--dataset_disk_path", default="")
        p.add_argument("--dataset_cache_dir", default="")
        p.add_argument("--seq_len", type=int, default=1024)
        p.add_argument("--router_prefix_tokens", type=int, default=256)
        p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
        p.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train_router")
    add_data_args(train)
    train.add_argument("--risk_label_file", required=True)
    train.add_argument("--router_input", choices=["raw_embedding", "prefix_hk_raw_attn"], default="prefix_hk_raw_attn")
    train.add_argument("--prefix_depth", type=int, default=4)
    train.add_argument("--skip_rate", type=float, default=0.25)
    train.add_argument("--skip_count", type=int, default=0)
    train.add_argument("--protected_head", type=int, default=4)
    train.add_argument("--protected_tail", type=int, default=2)
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--router_dim", type=int, default=256)
    train.add_argument("--router_heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--output_dir", required=True)
    train.set_defaults(func=train_router)

    eval_p = sub.add_parser("eval")
    add_data_args(eval_p)
    eval_p.add_argument("--eval_windows", type=int, default=512)
    eval_p.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    eval_p.add_argument("--sample_seed", type=int, default=42)
    eval_p.add_argument("--method", choices=["full", "static", "router"], required=True)
    eval_p.add_argument("--method_label", default="")
    eval_p.add_argument("--static_strategy", choices=C6_STATIC_STRATEGIES, default="uniform")
    eval_p.add_argument("--risk_router_ckpt", default="")
    eval_p.add_argument("--prefix_depth", type=int, default=4)
    eval_p.add_argument("--skip_rate", type=float, default=0.25)
    eval_p.add_argument("--skip_count", type=int, default=0)
    eval_p.add_argument("--protected_head", type=int, default=4)
    eval_p.add_argument("--protected_tail", type=int, default=2)
    eval_p.add_argument("--batch_size", type=int, default=1)
    eval_p.add_argument("--run_name", required=True)
    eval_p.add_argument("--output_json", required=True)
    eval_p.add_argument("--save_rows_in_json", action="store_true")
    eval_p.set_defaults(func=eval_method)

    diag = sub.add_parser("diagnose_overlap")
    add_data_args(diag)
    diag.add_argument("--risk_label_file", required=True)
    diag.add_argument("--risk_router_ckpt", required=True)
    diag.add_argument("--prefix_depth", type=int, default=4)
    diag.add_argument("--protected_head", type=int, default=4)
    diag.add_argument("--protected_tail", type=int, default=2)
    diag.add_argument("--max_samples", type=int, default=0)
    diag.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    diag.add_argument("--sample_seed", type=int, default=42)
    diag.add_argument("--batch_size", type=int, default=1)
    diag.add_argument("--output_jsonl", required=True)
    diag.add_argument("--summary_json", required=True)
    diag.set_defaults(func=diagnose_overlap)

    report = sub.add_parser("write_report")
    report.add_argument("--metric", action="append", default=[], help="NAME=PATH, e.g. Full=/tmp/full.json")
    report.add_argument("--label_metadata", default="")
    report.add_argument("--static_best_strategy", default="")
    report.add_argument("--static_best_val_json", default="")
    report.add_argument("--raw_training_metrics", default="")
    report.add_argument("--opal_training_metrics", default="")
    report.add_argument("--raw_overlap_summary", default="")
    report.add_argument("--opal_overlap_summary", default="")
    report.add_argument("--output_md", required=True)
    report.add_argument("--date", default="2026-06-02")
    report.set_defaults(func=write_report)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "method_label") and not args.method_label:
        if args.method == "full":
            args.method_label = "Full"
        elif args.method == "static":
            args.method_label = f"Static {args.static_strategy}"
        elif args.method == "router":
            args.method_label = "Router"
    if hasattr(args, "router_prefix_tokens") and int(args.router_prefix_tokens) >= int(args.seq_len):
        raise ValueError("--router_prefix_tokens must be smaller than --seq_len")
    args.func(args)


if __name__ == "__main__":
    main()
