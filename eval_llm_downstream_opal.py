#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)
sys.path.insert(0, current_dir)

from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2ForCausalLM

from eval_wikitext_opal_ppl import (
    load_candidate_router_checkpoint,
    load_ig_artifact,
    load_router_checkpoint,
    raw_embedding_state,
    router_forward,
    router_prefix_batch,
)
from wikitext_opal_utils import (
    RELATED_CANDIDATE_STRATEGIES,
    allowed_layers_from_policy,
    clear_custom_policy,
    detect_num_layers,
    dtype_from_precision,
    keep_masks_from_skip_risk,
    mask_key,
    related_candidate_masks,
    resolve_skip_budget,
    set_custom_policy,
    skipped_layers_from_keep_mask,
    static_keep_mask,
    summarize_keep_masks,
    write_json,
)


PRIMARY_TASKS = ("piqa", "openbookqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge")
METHOD_ORDER = (
    "Full",
    "Static ends_heavy",
    "Static best-on-val",
    "PuDDing-style",
    "IG-style",
    "layerwise_hidden_router",
    "Raw-SetBCE best-on-val",
    "OPAL-SetBCE best-on-val",
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def finite_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def load_dataset_split(path: str, name: Optional[str], split: str):
    kwargs = {"split": split}
    try:
        kwargs["trust_remote_code"] = True
        if name:
            return load_dataset(path, name, **kwargs)
        return load_dataset(path, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        if name:
            return load_dataset(path, name, **kwargs)
        return load_dataset(path, **kwargs)


def normalize_label(value) -> str:
    return str(value).strip()


def answer_index_from_key(answer_key, labels: Sequence[str]) -> Optional[int]:
    key = normalize_label(answer_key)
    label_map = {normalize_label(label): idx for idx, label in enumerate(labels)}
    if key in label_map:
        return label_map[key]
    if key.isdigit():
        digit = int(key)
        if 0 <= digit < len(labels):
            return digit
        if 1 <= digit <= len(labels):
            return digit - 1
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if key.upper() in alpha:
        idx = alpha.index(key.upper())
        if idx < len(labels):
            return idx
    return None


def task_examples(task: str, split: str) -> List[Dict[str, object]]:
    task = str(task)
    examples: List[Dict[str, object]] = []
    if task == "piqa":
        ds = load_dataset_split("piqa", None, split)
        for idx, row in enumerate(ds):
            gold = int(row["label"])
            examples.append(
                {
                    "id": f"piqa-{idx}",
                    "task": task,
                    "context": f"Question: {row['goal']}\nAnswer:",
                    "choices": [f" {row['sol1']}", f" {row['sol2']}"],
                    "gold": gold,
                }
            )
    elif task == "openbookqa":
        ds = load_dataset_split("openbookqa", "main", split)
        for idx, row in enumerate(ds):
            choice_text = [str(x) for x in row["choices"]["text"]]
            choice_labels = [str(x) for x in row["choices"]["label"]]
            gold = answer_index_from_key(row["answerKey"], choice_labels)
            if gold is None:
                continue
            examples.append(
                {
                    "id": f"openbookqa-{idx}",
                    "task": task,
                    "context": f"Question: {row['question_stem']}\nAnswer:",
                    "choices": [f" {text}" for text in choice_text],
                    "gold": int(gold),
                }
            )
    elif task == "winogrande":
        ds = load_dataset_split("winogrande", "winogrande_xl", split)
        for idx, row in enumerate(ds):
            sentence = str(row["sentence"])
            if "_" not in sentence:
                continue
            prefix, suffix = sentence.split("_", 1)
            choices = [f"{row['option1']}{suffix}", f"{row['option2']}{suffix}"]
            gold = int(row["answer"]) - 1
            examples.append(
                {
                    "id": f"winogrande-{idx}",
                    "task": task,
                    "context": prefix,
                    "choices": choices,
                    "gold": gold,
                }
            )
    elif task == "hellaswag":
        ds = load_dataset_split("hellaswag", None, split)
        for idx, row in enumerate(ds):
            ctx_a = str(row.get("ctx_a") or "")
            ctx_b = str(row.get("ctx_b") or "")
            context = (ctx_a + (" " + ctx_b.capitalize() if ctx_b else "")).strip()
            label = str(row.get("label", "")).strip()
            if not label.isdigit():
                continue
            examples.append(
                {
                    "id": f"hellaswag-{idx}",
                    "task": task,
                    "context": context,
                    "choices": [f" {ending}" for ending in row["endings"]],
                    "gold": int(label),
                }
            )
    elif task in {"arc_easy", "arc_challenge"}:
        config = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset_split("ai2_arc", config, split)
        for idx, row in enumerate(ds):
            choice_text = [str(x) for x in row["choices"]["text"]]
            choice_labels = [str(x) for x in row["choices"]["label"]]
            gold = answer_index_from_key(row["answerKey"], choice_labels)
            if gold is None:
                continue
            examples.append(
                {
                    "id": f"{task}-{idx}",
                    "task": task,
                    "context": f"Question: {row['question']}\nAnswer:",
                    "choices": [f" {text}" for text in choice_text],
                    "gold": int(gold),
                }
            )
    else:
        raise ValueError(f"Unsupported downstream task: {task}")
    return examples


def tokenize_example(tokenizer, example: Dict[str, object], max_length: int, router_prefix_tokens: int):
    context = str(example["context"])
    context_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    router_context_ids = list(context_ids[: max(1, int(router_prefix_tokens))])
    if not router_context_ids:
        router_context_ids = [tokenizer.eos_token_id]

    choice_rows = []
    for choice_idx, choice in enumerate(example["choices"]):
        cont_ids = tokenizer(str(choice), add_special_tokens=False)["input_ids"]
        if not cont_ids:
            cont_ids = [tokenizer.eos_token_id]
        ctx = list(context_ids)
        cont = list(cont_ids)
        if len(ctx) + len(cont) > int(max_length):
            overflow = len(ctx) + len(cont) - int(max_length)
            if overflow < len(ctx):
                ctx = ctx[overflow:]
            else:
                keep_cont = max(1, int(max_length))
                ctx = []
                cont = cont[-keep_cont:]
        input_ids = ctx + cont
        labels = [-100] * len(ctx) + cont
        choice_rows.append(
            {
                "choice_idx": int(choice_idx),
                "input_ids": input_ids,
                "labels": labels,
                "continuation_tokens": int(sum(1 for value in labels if int(value) != -100)),
            }
        )
    return {
        "id": example["id"],
        "task": example["task"],
        "gold": int(example["gold"]),
        "router_context_ids": router_context_ids,
        "choices": choice_rows,
        "num_choices": len(choice_rows),
    }


class MultipleChoiceDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, object]]):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[int(idx)]


def pad_rows(rows: Sequence[Sequence[int]], pad_value: int, device=None) -> torch.Tensor:
    max_len = max(1, max(len(row) for row in rows))
    out = torch.full((len(rows), max_len), int(pad_value), dtype=torch.long)
    for idx, row in enumerate(rows):
        values = list(row)
        out[idx, : len(values)] = torch.tensor(values, dtype=torch.long)
    return out.to(device) if device is not None else out


def attention_mask_from_rows(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    max_len = max(1, max(len(row) for row in rows))
    out = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, row in enumerate(rows):
        out[idx, : len(row)] = 1
    return out


def collate_examples(batch: Sequence[Dict[str, object]], pad_token_id: int):
    flat_input_ids = []
    flat_labels = []
    flat_example_index = []
    continuation_tokens = []
    choice_counts = []
    example_ids = []
    gold = []
    router_context_ids = []
    for example_index, example in enumerate(batch):
        example_ids.append(str(example["id"]))
        gold.append(int(example["gold"]))
        router_context_ids.append(list(example["router_context_ids"]))
        choices = list(example["choices"])
        choice_counts.append(len(choices))
        for choice in choices:
            flat_input_ids.append(list(choice["input_ids"]))
            flat_labels.append(list(choice["labels"]))
            flat_example_index.append(example_index)
            continuation_tokens.append(int(choice["continuation_tokens"]))
    input_ids = pad_rows(flat_input_ids, pad_token_id)
    labels = pad_rows(flat_labels, -100)
    attention_mask = attention_mask_from_rows(flat_input_ids)
    router_input_ids = pad_rows(router_context_ids, pad_token_id)
    router_attention_mask = attention_mask_from_rows(router_context_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "router_input_ids": router_input_ids,
        "router_attention_mask": router_attention_mask,
        "flat_example_index": torch.tensor(flat_example_index, dtype=torch.long),
        "choice_counts": torch.tensor(choice_counts, dtype=torch.long),
        "continuation_tokens": torch.tensor(continuation_tokens, dtype=torch.long),
        "gold": torch.tensor(gold, dtype=torch.long),
        "example_ids": example_ids,
    }


def sequence_loglikelihoods(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(safe_labels)
    total_nll = (losses * mask.float()).sum(dim=1)
    return -total_nll


def apply_layer_mask_and_forward(model, input_ids, attention_mask, layer_mask=None):
    if layer_mask is None:
        clear_custom_policy(model)
    else:
        set_custom_policy(model, layer_mask)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        clear_custom_policy(model)


def keep_masks_for_batch(args, method_state, model, batch, num_layers: int, skip_count: int, allowed_layers: Sequence[int]):
    method = args.method
    example_count = int(batch["router_input_ids"].size(0))
    selected_candidate_ids: List[Optional[str]] = [None for _ in range(example_count)]
    if method == "full":
        return [[1] * int(num_layers) for _ in range(example_count)], selected_candidate_ids
    if method == "static":
        static_mask = method_state["static_mask"]
        return [list(static_mask) for _ in range(example_count)], selected_candidate_ids
    if method == "router":
        router = method_state["router"]
        metadata = method_state["router_metadata"]
        pred_risk = router_forward(
            router,
            str(metadata.get("router_input") or "raw_embedding"),
            model,
            batch["router_input_ids"],
            batch["router_attention_mask"],
            num_layers,
            int(metadata.get("prefix_depth") or args.prefix_depth),
        )
        return keep_masks_from_skip_risk(pred_risk, skip_count=skip_count, allowed_layers=allowed_layers), selected_candidate_ids
    if method == "candidate_router":
        candidate_router = method_state["candidate_router"]
        metadata = method_state["candidate_router_metadata"]
        pred_delta = candidate_router(raw_embedding_state(model, batch["router_input_ids"], batch["router_attention_mask"]))
        candidate_indices = pred_delta.float().argmin(dim=1).detach().cpu().tolist()
        candidate_ids = [str(value) for value in metadata["candidate_ids"]]
        candidate_keep_masks = metadata["candidate_keep_masks"]
        keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
        selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        return keep_masks, selected_candidate_ids
    if method == "ig":
        ig_centers = method_state["ig_centers"]
        metadata = method_state["ig_metadata"]
        state = F.normalize(raw_embedding_state(model, batch["router_input_ids"], batch["router_attention_mask"]).float(), dim=-1)
        distances = torch.cdist(state.float(), ig_centers.float(), p=2)
        clusters = distances.argmin(dim=1).detach().cpu().tolist()
        cluster_candidate_indices = [int(idx) for idx in metadata["cluster_candidate_indices"]]
        candidate_ids = [str(value) for value in metadata["candidate_ids"]]
        candidate_keep_masks = metadata["candidate_keep_masks"]
        candidate_indices = [cluster_candidate_indices[int(cluster_idx)] for cluster_idx in clusters]
        keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
        selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        return keep_masks, selected_candidate_ids
    raise ValueError(f"Unsupported method: {method}")


def evaluate_task(args, accelerator, model, method_state, tokenizer, task: str, num_layers: int, skip_count: int, keep_count: int, allowed_layers: Sequence[int]):
    raw_examples = task_examples(task, args.split)
    if int(args.limit) > 0:
        raw_examples = raw_examples[: int(args.limit)]
    tokenized = [tokenize_example(tokenizer, row, args.max_length, args.router_prefix_tokens) for row in raw_examples]
    dataset = MultipleChoiceDataset(tokenized)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    dataloader = accelerator.prepare(dataloader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{args.run_name}_{task}.rank{accelerator.process_index}.jsonl"
    for stale in output_dir.glob(f"{args.run_name}_{task}.rank*.jsonl"):
        if accelerator.is_main_process:
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()

    iterator = tqdm(dataloader, desc=f"{args.method_label}-{task}") if accelerator.is_main_process else dataloader
    with shard_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in iterator:
            batch = {key: value.to(accelerator.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            example_keep_masks, selected_candidate_ids = keep_masks_for_batch(
                args,
                method_state,
                model,
                batch,
                num_layers,
                skip_count,
                allowed_layers,
            )
            flat_masks = [example_keep_masks[int(idx)] for idx in batch["flat_example_index"].detach().cpu().tolist()]
            layer_mask = None
            if args.method != "full":
                layer_mask = torch.tensor(flat_masks, dtype=torch.float32, device=accelerator.device)
            outputs = apply_layer_mask_and_forward(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                layer_mask=layer_mask,
            )
            loglik = sequence_loglikelihoods(outputs.logits, batch["labels"]).detach().cpu().tolist()
            cont_tokens = batch["continuation_tokens"].detach().cpu().tolist()
            choice_counts = batch["choice_counts"].detach().cpu().tolist()
            gold = batch["gold"].detach().cpu().tolist()
            example_ids = list(batch["example_ids"])

            cursor = 0
            for local_idx, choice_count in enumerate(choice_counts):
                scores = [float(v) for v in loglik[cursor : cursor + int(choice_count)]]
                lengths = [max(1, int(v)) for v in cont_tokens[cursor : cursor + int(choice_count)]]
                norm_scores = [score / length for score, length in zip(scores, lengths)]
                pred = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
                pred_norm = max(range(len(norm_scores)), key=lambda idx: (norm_scores[idx], -idx))
                keep_mask = example_keep_masks[local_idx]
                row = {
                    "id": example_ids[local_idx],
                    "task": task,
                    "gold": int(gold[local_idx]),
                    "pred": int(pred),
                    "pred_norm": int(pred_norm),
                    "acc": int(pred == int(gold[local_idx])),
                    "acc_norm": int(pred_norm == int(gold[local_idx])),
                    "scores": scores,
                    "scores_norm": norm_scores,
                    "continuation_tokens": lengths,
                    "keep_mask": [int(v) for v in keep_mask],
                    "mask_key": mask_key(keep_mask),
                    "skipped_layers": skipped_layers_from_keep_mask(keep_mask),
                }
                if selected_candidate_ids[local_idx] is not None:
                    row["selected_candidate_id"] = selected_candidate_ids[local_idx]
                f.write(json.dumps(row) + "\n")
                cursor += int(choice_count)

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None

    rows = []
    seen = set()
    for path in sorted(output_dir.glob(f"{args.run_name}_{task}.rank*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                rows.append(row)
    rows.sort(key=lambda row: row["id"])
    total = max(1, len(rows))
    mask_summary = summarize_keep_masks(
        [row["keep_mask"] for row in rows],
        expected_skip_count=0 if args.method == "full" else skip_count,
    )
    selected_candidate_distribution: Dict[str, int] = {}
    for row in rows:
        candidate_id = row.get("selected_candidate_id")
        if candidate_id:
            selected_candidate_distribution[str(candidate_id)] = selected_candidate_distribution.get(str(candidate_id), 0) + 1
    payload = {
        "task": task,
        "split": args.split,
        "method": args.method,
        "method_label": args.method_label,
        "seed": int(args.seed),
        "num_examples": int(len(rows)),
        "acc": float(sum(int(row["acc"]) for row in rows) / total),
        "acc_norm": float(sum(int(row["acc_norm"]) for row in rows) / total),
        "num_layers": int(num_layers),
        "skip_count": int(0 if args.method == "full" else skip_count),
        "keep_count": int(num_layers if args.method == "full" else keep_count),
        "protected_head": int(args.protected_head),
        "protected_tail": int(args.protected_tail),
        "average_kept_layers": float(mask_summary["average_kept_layers"]),
        "average_skipped_layers": float(mask_summary["average_skipped_layers"]),
        "exact_skip_count_rate": float(mask_summary["exact_skip_count_rate"]),
        "unique_masks": int(mask_summary["unique_masks"]),
        "selected_candidate_distribution": selected_candidate_distribution,
        "rows": rows if args.save_rows else [],
    }
    if args.method != "full" and float(payload["exact_skip_count_rate"]) < 1.0:
        raise ValueError(f"{args.run_name} {task} did not produce exact skip_count for every example.")
    return payload


def load_method_state(args, model, num_layers: int, hidden_size: int, device, skip_count: int, allowed_layers: Sequence[int]):
    state = {}
    if args.method == "static":
        state["static_mask"] = static_keep_mask(
            args.static_strategy,
            num_layers=num_layers,
            skip_count=skip_count,
            protected_head=args.protected_head,
            protected_tail=args.protected_tail,
            seed=args.seed,
        )
    elif args.method == "router":
        router, metadata = load_router_checkpoint(args.risk_router_ckpt, hidden_size, num_layers, device)
        state["router"] = router
        state["router_metadata"] = metadata
    elif args.method == "candidate_router":
        candidate_router, metadata = load_candidate_router_checkpoint(args.candidate_router_ckpt, hidden_size, device)
        state["candidate_router"] = candidate_router
        state["candidate_router_metadata"] = metadata
    elif args.method == "ig":
        centers, metadata = load_ig_artifact(args.ig_artifact, device)
        state["ig_centers"] = centers
        state["ig_metadata"] = metadata
    return state


def eval_many(args):
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype_from_precision(args.precision))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    skip_count, keep_count = resolve_skip_budget(num_layers, args.skip_rate, args.skip_count)
    allowed_layers = allowed_layers_from_policy(num_layers, args.protected_head, args.protected_tail)
    method_state = load_method_state(args, model, num_layers, hidden_size, device, skip_count, allowed_layers)

    tasks = parse_csv(args.tasks)
    task_metrics = []
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "method_label": args.method_label,
                    "tasks": tasks,
                    "seed": int(args.seed),
                    "max_length": int(args.max_length),
                    "router_prefix_tokens": int(args.router_prefix_tokens),
                    "skip_count": int(0 if args.method == "full" else skip_count),
                    "target_leakage_guard": "routers see context only; candidate continuation tokens are never used for mask selection",
                },
                indent=2,
            )
        )

    for task in tasks:
        metric = evaluate_task(args, accelerator, model, method_state, tokenizer, task, num_layers, skip_count, keep_count, allowed_layers)
        if accelerator.is_main_process and metric is not None:
            task_metrics.append(metric)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        acc_values = [float(row["acc"]) for row in task_metrics]
        acc_norm_values = [float(row["acc_norm"]) for row in task_metrics]
        selected_distribution: Dict[str, int] = {}
        for metric in task_metrics:
            for key, value in metric.get("selected_candidate_distribution", {}).items():
                selected_distribution[str(key)] = selected_distribution.get(str(key), 0) + int(value)
        payload = {
            "run_name": args.run_name,
            "method": args.method,
            "method_label": args.method_label,
            "seed": int(args.seed),
            "model": args.model,
            "tasks": tasks,
            "split": args.split,
            "max_length": int(args.max_length),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "skip_rate": float(args.skip_rate),
            "skip_count": int(0 if args.method == "full" else skip_count),
            "protected_head": int(args.protected_head),
            "protected_tail": int(args.protected_tail),
            "precision": args.precision,
            "batch_size": int(args.batch_size),
            "task_metrics": task_metrics,
            "average_acc": finite_mean(acc_values),
            "average_acc_norm": finite_mean(acc_norm_values),
            "unique_masks_mean": finite_mean([float(row["unique_masks"]) for row in task_metrics]),
            "exact_skip_count_rate_mean": finite_mean([float(row["exact_skip_count_rate"]) for row in task_metrics]),
            "average_kept_layers_mean": finite_mean([float(row["average_kept_layers"]) for row in task_metrics]),
            "selected_candidate_distribution": selected_distribution,
            "uses_greedy_labels_at_eval": False,
            "compensation": "none",
            "target_leakage_guard": "routers see context only; candidate continuation tokens are never used for mask selection",
            "mask_application": "config.custom_layer_mask",
        }
        write_json(args.output_json, payload)
        print(
            json.dumps(
                {
                    "run_name": payload["run_name"],
                    "method_label": payload["method_label"],
                    "seed": payload["seed"],
                    "average_acc": payload["average_acc"],
                    "average_acc_norm": payload["average_acc_norm"],
                    "unique_masks_mean": payload["unique_masks_mean"],
                },
                indent=2,
            )
        )


def load_metric(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def method_path(output_root: Path, seed: int, method_slug: str) -> Path:
    return output_root / "metrics" / f"seed{int(seed)}" / f"{method_slug}.json"


def task_metric_by_name(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {str(row["task"]): row for row in payload.get("task_metrics", [])}


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def summarize(args):
    output_root = Path(args.output_root)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    methods = parse_csv(args.methods)
    tasks = parse_csv(args.tasks)
    rows = []
    missing = []
    by_seed_method: Dict[Tuple[int, str], Dict[str, object]] = {}
    for seed in seeds:
        for method_slug in methods:
            path = method_path(output_root, seed, method_slug)
            payload = load_metric(path)
            if payload is None:
                missing.append(str(path))
                continue
            by_seed_method[(seed, str(payload["method_label"]))] = payload
            task_map = task_metric_by_name(payload)
            for task in tasks:
                metric = task_map.get(task)
                if not metric:
                    missing.append(f"{path}:{task}")
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "method": str(payload["method_label"]),
                        "task": task,
                        "acc": float(metric["acc"]),
                        "acc_norm": float(metric["acc_norm"]),
                        "unique_masks": int(metric["unique_masks"]),
                        "exact_skip_count_rate": float(metric["exact_skip_count_rate"]),
                        "average_kept_layers": float(metric["average_kept_layers"]),
                        "selected_candidate_distribution": metric.get("selected_candidate_distribution", {}),
                    }
                )

    full_by_seed_task = {
        (row["seed"], row["task"]): row
        for row in rows
        if row["method"] == "Full"
    }
    for row in rows:
        full = full_by_seed_task.get((row["seed"], row["task"]))
        row["retention_acc"] = float(row["acc"] / full["acc"]) if full and float(full["acc"]) > 0 else float("nan")
        row["retention_acc_norm"] = (
            float(row["acc_norm"] / full["acc_norm"]) if full and float(full["acc_norm"]) > 0 else float("nan")
        )

    lines = []
    lines.append("# LLM Eval Downstream Results")
    lines.append("")
    lines.append(f"Last updated: {args.date}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- tasks: `{tasks}`")
    lines.append(f"- seeds: `{seeds}`")
    lines.append(f"- max_length: `{args.max_length}`")
    lines.append(f"- router_prefix_tokens: `{args.router_prefix_tokens}`")
    lines.append(f"- skip_rate / skip_count: `{args.skip_rate}` / `{args.skip_count}`")
    lines.append(f"- protected_head / protected_tail: `{args.protected_head}` / `{args.protected_tail}`")
    lines.append("- compensation: `none`")
    lines.append("- mask selection leakage guard: routers see context only; candidate answer continuation tokens are never used for mask selection")
    lines.append("")
    lines.append("## Per-Seed Task Results")
    lines.append("")
    lines.append("| seed | task | method | acc | acc_norm | retention_acc | retention_acc_norm | unique_masks | exact_skip_count_rate | average_kept_layers | selected candidates |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for seed in seeds:
        for task in tasks:
            for method in METHOD_ORDER:
                match = next((row for row in rows if row["seed"] == seed and row["task"] == task and row["method"] == method), None)
                if not match:
                    continue
                selected = match.get("selected_candidate_distribution") or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(seed),
                            task,
                            method,
                            fmt(match["acc"]),
                            fmt(match["acc_norm"]),
                            fmt(match["retention_acc"]),
                            fmt(match["retention_acc_norm"]),
                            str(match["unique_masks"]),
                            fmt(match["exact_skip_count_rate"], 3),
                            fmt(match["average_kept_layers"], 2),
                            f"`{selected}`" if selected else "{}",
                        ]
                    )
                    + " |"
                )

    lines.append("")
    lines.append("## Three-Seed Mean/Std")
    lines.append("")
    lines.append("| task | method | acc mean | acc std | acc_norm mean | acc_norm std | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean | average_kept_layers mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for task in tasks:
        for method in METHOD_ORDER:
            vals = [row for row in rows if row["task"] == task and row["method"] == method]
            if not vals:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        task,
                        method,
                        fmt(finite_mean([v["acc"] for v in vals])),
                        fmt(finite_std([v["acc"] for v in vals])),
                        fmt(finite_mean([v["acc_norm"] for v in vals])),
                        fmt(finite_std([v["acc_norm"] for v in vals])),
                        fmt(finite_mean([v["retention_acc"] for v in vals])),
                        fmt(finite_mean([v["retention_acc_norm"] for v in vals])),
                        fmt(finite_mean([v["unique_masks"] for v in vals])),
                        fmt(finite_mean([v["exact_skip_count_rate"] for v in vals]), 3),
                        fmt(finite_mean([v["average_kept_layers"] for v in vals]), 2),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Six-Task Average")
    lines.append("")
    lines.append("| method | acc mean | acc std over seeds | acc_norm mean | acc_norm std over seeds | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    six_task_by_seed: Dict[str, List[float]] = defaultdict(list)
    six_task_norm_by_seed: Dict[str, List[float]] = defaultdict(list)
    for method in METHOD_ORDER:
        seed_acc = []
        seed_acc_norm = []
        seed_ret = []
        seed_ret_norm = []
        seed_unique = []
        seed_exact = []
        for seed in seeds:
            vals = [row for row in rows if row["seed"] == seed and row["method"] == method and row["task"] in tasks]
            if not vals:
                continue
            seed_acc.append(finite_mean([v["acc"] for v in vals]))
            seed_acc_norm.append(finite_mean([v["acc_norm"] for v in vals]))
            seed_ret.append(finite_mean([v["retention_acc"] for v in vals]))
            seed_ret_norm.append(finite_mean([v["retention_acc_norm"] for v in vals]))
            seed_unique.append(finite_mean([v["unique_masks"] for v in vals]))
            seed_exact.append(finite_mean([v["exact_skip_count_rate"] for v in vals]))
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(finite_mean(seed_acc)),
                    fmt(finite_std(seed_acc)),
                    fmt(finite_mean(seed_acc_norm)),
                    fmt(finite_std(seed_acc_norm)),
                    fmt(finite_mean(seed_ret)),
                    fmt(finite_mean(seed_ret_norm)),
                    fmt(finite_mean(seed_unique)),
                    fmt(finite_mean(seed_exact), 3),
                ]
            )
            + " |"
        )
        six_task_by_seed[method] = seed_acc
        six_task_norm_by_seed[method] = seed_acc_norm

    def mean_for(method: str, metric: str = "acc_norm") -> float:
        values = six_task_norm_by_seed[method] if metric == "acc_norm" else six_task_by_seed[method]
        return finite_mean(values)

    opal_norm = mean_for("OPAL-SetBCE best-on-val", "acc_norm")
    comparisons = {
        "Raw-SetBCE best-on-val": opal_norm > mean_for("Raw-SetBCE best-on-val", "acc_norm"),
        "layerwise_hidden_router": opal_norm > mean_for("layerwise_hidden_router", "acc_norm"),
        "PuDDing-style": opal_norm > mean_for("PuDDing-style", "acc_norm"),
        "IG-style": opal_norm > mean_for("IG-style", "acc_norm"),
        "Static best-on-val": opal_norm > mean_for("Static best-on-val", "acc_norm"),
        "Static ends_heavy": opal_norm > mean_for("Static ends_heavy", "acc_norm"),
    }

    lines.append("")
    lines.append("## Required Judgments")
    lines.append("")
    for method, wins in comparisons.items():
        lines.append(f"- OPAL best-on-val beats `{method}` by six-task mean acc_norm: `{wins}`")
    opal_unique = [
        finite_mean([row["unique_masks"] for row in rows if row["seed"] == seed and row["method"] == "OPAL-SetBCE best-on-val"])
        for seed in seeds
    ]
    lines.append(f"- OPAL best-on-val unique_masks mean per seed: `{[round(v, 4) for v in opal_unique]}`")

    for method in ("PuDDing-style", "IG-style"):
        distributions = []
        for seed in seeds:
            payload = by_seed_method.get((seed, method))
            if payload:
                distributions.append((seed, payload.get("selected_candidate_distribution", {})))
        collapsed = all(
            isinstance(dist, dict) and len(dist) == 1 and "ends_heavy" in dist
            for _, dist in distributions
        )
        lines.append(f"- {method} candidate distributions by seed: `{distributions}`")
        lines.append(f"- {method} degenerates to ends_heavy only: `{collapsed}`")

    if any(v <= 2 for v in opal_unique if math.isfinite(v)):
        lines.append("- Caveat: OPAL best-on-val unique_masks is very low on at least one seed; write the downstream result with the same static-like checkpoint caveat as WikiText-2.")
    if missing:
        lines.append("")
        lines.append("## Missing Artifacts")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item}`")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="OPAL layer-skip downstream multiple-choice evaluator.")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--tasks", default=",".join(PRIMARY_TASKS))
    eval_p.add_argument("--split", default="validation")
    eval_p.add_argument("--method", choices=["full", "static", "router", "candidate_router", "ig"], required=True)
    eval_p.add_argument("--method_label", required=True)
    eval_p.add_argument("--static_strategy", default="ends_heavy")
    eval_p.add_argument("--risk_router_ckpt", default="")
    eval_p.add_argument("--candidate_router_ckpt", default="")
    eval_p.add_argument("--ig_artifact", default="")
    eval_p.add_argument("--skip_rate", type=float, default=0.25)
    eval_p.add_argument("--skip_count", type=int, default=7)
    eval_p.add_argument("--protected_head", type=int, default=4)
    eval_p.add_argument("--protected_tail", type=int, default=2)
    eval_p.add_argument("--prefix_depth", type=int, default=4)
    eval_p.add_argument("--router_prefix_tokens", type=int, default=256)
    eval_p.add_argument("--max_length", type=int, default=1024)
    eval_p.add_argument("--batch_size", type=int, default=8)
    eval_p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--limit", type=int, default=0)
    eval_p.add_argument("--run_name", required=True)
    eval_p.add_argument("--output_dir", required=True)
    eval_p.add_argument("--output_json", required=True)
    eval_p.add_argument("--save_rows", action="store_true")
    eval_p.set_defaults(func=eval_many)

    report = sub.add_parser("summarize")
    report.add_argument("--output_root", required=True)
    report.add_argument("--output_md", required=True)
    report.add_argument("--model", default="/workspace/ckpts/Qwen2.5-1.5B")
    report.add_argument("--tasks", default=",".join(PRIMARY_TASKS))
    report.add_argument("--methods", default="full,static_ends_heavy,static_best_on_val,pudding,ig,layerwise,raw_best_val,opal_best_val")
    report.add_argument("--seeds", default="42,13,3407")
    report.add_argument("--max_length", type=int, default=1024)
    report.add_argument("--router_prefix_tokens", type=int, default=256)
    report.add_argument("--skip_rate", type=float, default=0.25)
    report.add_argument("--skip_count", type=int, default=7)
    report.add_argument("--protected_head", type=int, default=4)
    report.add_argument("--protected_tail", type=int, default=2)
    report.add_argument("--date", default="2026-06-02")
    report.set_defaults(func=summarize)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
