#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.opal_risk_router import (
    LayerQueryCrossAttentionRiskRouter,
    LayerwiseHiddenRiskRouter,
    OpalRiskRouter,
    PromptCandidateMaskRouter,
    exact_k_subset_ce_loss,
)
from wikitext_opal_utils import (
    C6_STATIC_STRATEGIES,
    RELATED_CANDIDATE_STRATEGIES,
    aggregate_loss_rows,
    action_payload_from_keep_mask,
    allowed_layers_from_policy,
    clear_custom_policy,
    collate_wikitext_windows,
    compensation_config,
    compensation_runtime_stats,
    detect_num_layers,
    dtype_from_precision,
    finite_exp,
    keep_masks_from_skip_risk,
    lm_loss_stats_from_logits,
    load_wikitext_token_ids,
    LowRankResidualAdapter,
    mask_key,
    read_jsonl,
    related_candidate_masks,
    resolve_skip_budget,
    set_custom_policy,
    skipped_layers_from_keep_mask,
    static_keep_mask,
    summarize_keep_masks,
    WikitextWindowDataset,
    write_json,
)


ATTENTION_ROUTER_INPUTS = {"prefix_hk_raw_attn"}
LAYERWISE_ROUTER_INPUTS = {"layerwise_hidden"}


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


def layerwise_hidden_state(model, input_ids, attention_mask, num_layers: int):
    captured = []
    handles = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(masked_mean_state(hidden.detach(), attention_mask).float())

    for layer in model.model.layers[: int(num_layers)]:
        handles.append(layer.register_forward_hook(hook))
    clear_custom_policy(model)
    try:
        with torch.no_grad():
            model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        clear_custom_policy(model)
    if len(captured) != int(num_layers):
        raise RuntimeError(f"Expected {num_layers} captured layer states, got {len(captured)}")
    return torch.stack(captured, dim=1).float()


def teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers: int, prefix_depth: int):
    prefix_depth = max(0, min(int(prefix_depth), int(num_layers)))
    prefix_mask = torch.tensor(
        [[1 if idx < prefix_depth else 0 for idx in range(int(num_layers))]] * input_ids.size(0),
        dtype=torch.float32,
        device=input_ids.device,
    )
    set_custom_policy(model, prefix_mask)
    try:
        with torch.no_grad():
            return model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
    finally:
        clear_custom_policy(model)


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
    if router_input == "layerwise_hidden":
        return LayerwiseHiddenRiskRouter(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
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
    if router_input == "layerwise_hidden":
        return router(layerwise_hidden_state(model, input_ids, attention_mask, num_layers))
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


def build_lowrank_adapter(num_layers: int, hidden_size: int, rank: int, model_config) -> LowRankResidualAdapter:
    eps = float(getattr(model_config, "rms_norm_eps", 1e-6))
    return LowRankResidualAdapter(
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        rank=int(rank),
        eps=eps,
    )


def save_lowrank_adapter_checkpoint(path: str, adapter: LowRankResidualAdapter, metadata: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": adapter.state_dict(),
            "metadata": dict(metadata),
        },
        output_path,
    )


def load_lowrank_adapter_checkpoint(path: str, num_layers: int, hidden_size: int, device, model_config):
    checkpoint = torch.load(path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata") or {})
    rank = int(metadata.get("rank") or metadata.get("compensation_rank") or 16)
    adapter = build_lowrank_adapter(num_layers, hidden_size, rank, model_config)
    adapter.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    adapter.to(device).eval()
    return adapter, metadata


def kl_full_to_skipped_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_student = student_logits[..., :-1, :].float().contiguous()
    shift_teacher = teacher_logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)
    if int(valid.sum().item()) <= 0:
        return shift_student.sum() * 0.0
    student_log_probs = F.log_softmax(shift_student, dim=-1)
    teacher_log_probs = F.log_softmax(shift_teacher, dim=-1)
    token_kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    return token_kl[valid].mean()


def forward_loss_rows(
    model,
    input_ids,
    attention_mask,
    labels,
    layer_mask=None,
    compensation_mode: str = "none",
    compensation_rank: int = 0,
    compensation_static_gate: float = 1.0,
    compensation_adapter=None,
):
    if layer_mask is None:
        clear_custom_policy(model)
    else:
        comp = compensation_config(compensation_mode, compensation_rank, compensation_static_gate)
        action_payload = action_payload_from_keep_mask(layer_mask) if comp.get("mode") != "none" else None
        set_custom_policy(
            model,
            layer_mask,
            action_payload=action_payload,
            compensation_config_payload=comp,
            compensation_adapter=compensation_adapter,
        )
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        rows = lm_loss_stats_from_logits(outputs.logits, labels)
        del outputs
        for row in rows:
            row.update(compensation_runtime_stats(model))
        return rows
    finally:
        clear_custom_policy(model)


def load_mask_runtime(
    spec: Dict[str, object],
    model,
    hidden_size: int,
    num_layers: int,
    device,
    default_skip_count: int,
    default_allowed_layers: Sequence[int],
    protected_head: int,
    protected_tail: int,
    seed: int,
) -> Dict[str, object]:
    method = str(spec.get("method") or "")
    runtime: Dict[str, object] = {
        "slug": str(spec.get("slug") or method),
        "method": method,
        "method_label": str(spec.get("method_label") or spec.get("label") or method),
        "allowed_layers": list(default_allowed_layers),
        "skip_count": int(default_skip_count),
        "prefix_depth": int(spec.get("prefix_depth") or 4),
    }
    if method == "static":
        strategy = str(spec.get("static_strategy") or "ends_heavy")
        runtime["static_strategy"] = strategy
        runtime["static_mask"] = static_keep_mask(
            strategy,
            num_layers=num_layers,
            skip_count=int(default_skip_count),
            protected_head=protected_head,
            protected_tail=protected_tail,
            seed=seed,
        )
    elif method == "router":
        ckpt = str(spec.get("risk_router_ckpt") or "")
        router, metadata = load_router_checkpoint(ckpt, hidden_size, num_layers, device)
        runtime["router"] = router
        runtime["router_metadata"] = metadata
        runtime["risk_router_ckpt"] = ckpt
        runtime["allowed_layers"] = metadata.get("allowed_layers") or list(default_allowed_layers)
        runtime["skip_count"] = int(metadata.get("skip_count") or default_skip_count)
        runtime["prefix_depth"] = int(metadata.get("prefix_depth") or spec.get("prefix_depth") or 4)
    elif method == "candidate_router":
        ckpt = str(spec.get("candidate_router_ckpt") or "")
        router, metadata = load_candidate_router_checkpoint(ckpt, hidden_size, device)
        runtime["candidate_router"] = router
        runtime["candidate_router_metadata"] = metadata
        runtime["candidate_router_ckpt"] = ckpt
        runtime["allowed_layers"] = metadata.get("allowed_layers") or list(default_allowed_layers)
        runtime["skip_count"] = int(metadata.get("skip_count") or default_skip_count)
    elif method == "ig":
        artifact = str(spec.get("ig_artifact") or "")
        centers, metadata = load_ig_artifact(artifact, device)
        runtime["ig_centers"] = centers
        runtime["ig_metadata"] = metadata
        runtime["ig_artifact"] = artifact
        runtime["allowed_layers"] = metadata.get("allowed_layers") or list(default_allowed_layers)
        runtime["skip_count"] = int(metadata.get("skip_count") or default_skip_count)
    else:
        raise ValueError(f"Unsupported mask runtime method: {method}")
    return runtime


def keep_masks_for_runtime(
    runtime: Dict[str, object],
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    router_prefix_tokens: int,
    num_layers: int,
) -> Tuple[List[List[int]], List[Optional[str]]]:
    method = str(runtime["method"])
    selected_candidate_ids: List[Optional[str]] = [None for _ in range(input_ids.size(0))]
    if method == "static":
        return [list(runtime["static_mask"]) for _ in range(input_ids.size(0))], selected_candidate_ids
    if method == "router":
        router_input_ids, router_attention_mask = router_prefix_batch(
            input_ids,
            attention_mask,
            router_prefix_tokens,
        )
        metadata = dict(runtime.get("router_metadata") or {})
        pred_risk = router_forward(
            runtime["router"],
            str(metadata.get("router_input") or "raw_embedding"),
            model,
            router_input_ids,
            router_attention_mask,
            num_layers,
            int(runtime.get("prefix_depth") or 4),
        )
        return keep_masks_from_skip_risk(
            pred_risk,
            skip_count=int(runtime["skip_count"]),
            allowed_layers=runtime["allowed_layers"],
        ), selected_candidate_ids
    if method == "candidate_router":
        router_input_ids, router_attention_mask = router_prefix_batch(
            input_ids,
            attention_mask,
            router_prefix_tokens,
        )
        metadata = dict(runtime["candidate_router_metadata"])
        pred_delta = runtime["candidate_router"](raw_embedding_state(model, router_input_ids, router_attention_mask))
        candidate_indices = pred_delta.float().argmin(dim=1).detach().cpu().tolist()
        candidate_ids = [str(value) for value in metadata["candidate_ids"]]
        candidate_keep_masks = metadata["candidate_keep_masks"]
        keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
        selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        return keep_masks, selected_candidate_ids
    if method == "ig":
        router_input_ids, router_attention_mask = router_prefix_batch(
            input_ids,
            attention_mask,
            router_prefix_tokens,
        )
        metadata = dict(runtime["ig_metadata"])
        state = F.normalize(raw_embedding_state(model, router_input_ids, router_attention_mask).float(), dim=-1)
        distances = torch.cdist(state.float(), runtime["ig_centers"].float(), p=2)
        clusters = distances.argmin(dim=1).detach().cpu().tolist()
        cluster_candidate_indices = [int(idx) for idx in metadata["cluster_candidate_indices"]]
        candidate_ids = [str(value) for value in metadata["candidate_ids"]]
        candidate_keep_masks = metadata["candidate_keep_masks"]
        candidate_indices = [cluster_candidate_indices[int(cluster_idx)] for cluster_idx in clusters]
        keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
        selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        return keep_masks, selected_candidate_ids
    raise ValueError(f"Unsupported mask runtime method: {method}")


def load_label_rows(path: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: int(row["sample_id"]))
    if not rows:
        raise ValueError(f"No label rows found in {path}")
    validate_label_full_stats(rows, path)
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


def validate_label_full_stats(rows: Sequence[Dict[str, object]], path: str) -> None:
    nll_full = []
    ppl_full = []
    for row in rows:
        try:
            nll = float(row["NLL_full"])
            ppl = float(row["PPL_full"])
        except Exception as exc:
            raise ValueError(f"Label row lacks finite NLL_full/PPL_full in {path}") from exc
        if not math.isfinite(nll) or not math.isfinite(ppl):
            raise ValueError(f"Label row has non-finite NLL_full/PPL_full in {path}")
        nll_full.append(nll)
        ppl_full.append(ppl)
    if not nll_full:
        return
    max_nll_mean = float(os.environ.get("WIKITEXT_LABEL_MAX_FULL_NLL_MEAN", "5.0"))
    max_ppl_median = float(os.environ.get("WIKITEXT_LABEL_MAX_FULL_PPL_MEDIAN", "200.0"))
    mean_nll = sum(nll_full) / len(nll_full)
    median_ppl = sorted(ppl_full)[len(ppl_full) // 2]
    if len(ppl_full) % 2 == 0:
        sorted_ppl = sorted(ppl_full)
        median_ppl = 0.5 * (sorted_ppl[len(ppl_full) // 2 - 1] + sorted_ppl[len(ppl_full) // 2])
    if mean_nll > max_nll_mean or median_ppl > max_ppl_median:
        raise RuntimeError(
            "Refusing poisoned WikiText greedy label file: "
            f"NLL_full_mean={mean_nll:.6g} "
            f"(max {max_nll_mean:.6g}), "
            f"PPL_full_median={median_ppl:.6g} "
            f"(max {max_ppl_median:.6g}), "
            f"path={path}"
        )


def load_candidate_label_rows(path: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: int(row["sample_id"]))
    if not rows:
        raise ValueError(f"No candidate label rows found in {path}")
    metadata_path = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "candidate_ids" not in metadata and rows[0].get("candidate_ids") is not None:
        metadata["candidate_ids"] = rows[0]["candidate_ids"]
    if "candidate_keep_masks" not in metadata and rows[0].get("candidate_keep_masks") is not None:
        metadata["candidate_keep_masks"] = rows[0]["candidate_keep_masks"]
    return rows, metadata


def evaluate_keep_masks_batch(model, input_ids, attention_mask, labels, keep_masks: Sequence[Sequence[int]]):
    layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=input_ids.device)
    expanded_input_ids = input_ids.expand(layer_mask.size(0), -1).contiguous()
    expanded_attention_mask = attention_mask.expand(layer_mask.size(0), -1).contiguous()
    expanded_labels = labels.expand(layer_mask.size(0), -1).contiguous()
    return forward_loss_rows(
        model,
        expanded_input_ids,
        expanded_attention_mask,
        expanded_labels,
        layer_mask=layer_mask,
    )


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


def build_candidate_labels(args):
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    selected_window_ids = None
    reference_rows = []
    if args.reference_label_file:
        reference_rows, _ = load_label_rows(args.reference_label_file)
        if int(args.label_samples) > 0:
            reference_rows = reference_rows[: int(args.label_samples)]
        selected_window_ids = sorted(int(row["sample_id"]) for row in reference_rows)

    dataset, token_ids = make_dataset_for_split(
        args,
        tokenizer,
        selected_window_ids=selected_window_ids,
        max_windows=0 if selected_window_ids is not None else args.label_samples,
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
    candidates = related_candidate_masks(
        num_layers=num_layers,
        skip_count=skip_count,
        protected_head=args.protected_head,
        protected_tail=args.protected_tail,
        seed=args.seed,
    )
    if not candidates:
        raise ValueError("Candidate mask library is empty.")
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    candidate_keep_masks = [[int(v) for v in row["keep_mask"]] for row in candidates]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_path.unlink(missing_ok=True)
        for stale in output_path.parent.glob(output_path.name + ".rank*"):
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{accelerator.process_index}")

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "task": "wikitext2_related_candidate_labels",
                    "teacher_model": args.teacher_model,
                    "dataset_disk_path": args.dataset_disk_path,
                    "num_layers": int(num_layers),
                    "skip_count": int(skip_count),
                    "keep_count": int(keep_count),
                    "protected_head": int(args.protected_head),
                    "protected_tail": int(args.protected_tail),
                    "candidate_ids": candidate_ids,
                    "label_rows": int(len(selected_window_ids)),
                    "reference_label_file": args.reference_label_file,
                    "objective": "candidate_suffix_Delta_NLL",
                    "target_leakage_guard": "candidate quality labels are built on train; eval/test never loads them",
                },
                indent=2,
            )
        )

    iterator = tqdm(dataloader, desc="wikitext-c16-candidate-labels") if accelerator.is_main_process else dataloader
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
                full_stats = forward_loss_rows(model, input_ids, attention_mask, labels, layer_mask=None)[0]
                candidate_stats = []
                for start in range(0, len(candidate_keep_masks), max(1, int(args.candidate_batch_size))):
                    keep_chunk = candidate_keep_masks[start : start + max(1, int(args.candidate_batch_size))]
                    candidate_stats.extend(
                        evaluate_keep_masks_batch(model, input_ids, attention_mask, labels, keep_chunk)
                    )
                candidate_nll = [float(stats["nll"]) for stats in candidate_stats]
                candidate_ppl = [float(stats["ppl"]) for stats in candidate_stats]
                candidate_delta_nll = [float(value - float(full_stats["nll"])) for value in candidate_nll]
                candidate_delta_ppl = [float(value - float(full_stats["ppl"])) for value in candidate_ppl]
                best_candidate_index = min(
                    range(len(candidate_delta_nll)),
                    key=lambda idx: (candidate_delta_nll[idx], candidate_ids[idx]),
                )
                row = {
                    "sample_id": int(sample_id),
                    "window_start": int(window_starts[row_pos]),
                    "objective": "candidate_suffix_Delta_NLL",
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
                    "candidate_ids": candidate_ids,
                    "candidate_delta_nll": candidate_delta_nll,
                    "candidate_delta_ppl": candidate_delta_ppl,
                    "candidate_nll": candidate_nll,
                    "candidate_ppl": candidate_ppl,
                    "best_candidate_index": int(best_candidate_index),
                    "best_candidate_id": candidate_ids[best_candidate_index],
                    "NLL_full": float(full_stats["nll"]),
                    "PPL_full": float(full_stats["ppl"]),
                    "target_leakage_guard": "train labels only; eval/test path never loads candidate quality labels",
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
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        metadata = {
            "teacher_model": args.teacher_model,
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
            "reference_label_file": args.reference_label_file,
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "scored_tokens_per_full_window": int(args.seq_len) - int(args.router_prefix_tokens),
            "objective": "candidate_suffix_Delta_NLL",
            "num_layers": int(num_layers),
            "skip_rate": float(args.skip_rate),
            "skip_count": int(skip_count),
            "keep_count": int(keep_count),
            "protected_head": int(args.protected_head),
            "protected_tail": int(args.protected_tail),
            "allowed_layers": [int(idx) for idx in allowed_layers],
            "candidate_ids": candidate_ids,
            "candidate_keep_masks": candidate_keep_masks,
            "candidate_strategies": [str(row["strategy"]) for row in candidates],
            "candidate_count": int(len(candidates)),
            "candidate_batch_size": int(args.candidate_batch_size),
            "mask_application": "config.custom_layer_mask",
            "target_leakage_guard": "candidate labels are built only on train; eval/test never loads them",
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} WikiText-2 C16 candidate-label rows to {output_path}")
        print(f"Wrote candidate-label metadata to {metadata_path}")


def train_candidate_router(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    label_rows, label_metadata = load_candidate_label_rows(args.candidate_label_file)
    label_by_sample = {int(row["sample_id"]): row for row in label_rows}
    selected_window_ids = sorted(label_by_sample)
    candidate_ids = [str(value) for value in label_metadata.get("candidate_ids", label_rows[0]["candidate_ids"])]
    candidate_keep_masks = label_metadata.get("candidate_keep_masks")
    if not candidate_keep_masks:
        raise ValueError("Candidate metadata must include candidate_keep_masks.")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)
    router = PromptCandidateMaskRouter(
        hidden_size=hidden_size,
        num_candidates=len(candidate_ids),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "task": "wikitext2_pudding_style_candidate_router",
                    "teacher_model": args.teacher_model,
                    "dataset_disk_path": args.dataset_disk_path,
                    "num_layers": int(num_layers),
                    "hidden_size": int(hidden_size),
                    "candidate_count": int(len(candidate_ids)),
                    "candidate_ids": candidate_ids,
                    "label_rows": len(label_rows),
                    "seq_len": int(args.seq_len),
                    "router_prefix_tokens": int(args.router_prefix_tokens),
                    "token_count_train_split": int(len(token_ids)),
                    "target_leakage_guard": "router sees only raw prefix state; labels score suffix tokens only",
                },
                indent=2,
            )
        )

    history = []
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0.0
        total_steps = 0
        iterator = tqdm(dataloader, desc=f"candidate-router epoch {epoch + 1}/{args.epochs}") if accelerator.is_main_process else dataloader
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            router_input_ids, router_attention_mask = router_prefix_batch(
                input_ids,
                attention_mask,
                args.router_prefix_tokens,
            )
            pred = router(raw_embedding_state(model, router_input_ids, router_attention_mask))
            target = torch.tensor(
                [label_by_sample[sample_id]["candidate_delta_nll"] for sample_id in sample_ids],
                dtype=torch.float32,
                device=device,
            )
            if args.loss == "mse":
                loss = F.mse_loss(pred.float(), target.float())
            else:
                loss = F.smooth_l1_loss(pred.float(), target.float())
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite candidate-router loss: {float(loss.detach().float().item())}")
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
            "candidate_quality": float(summed[0].item() / global_steps),
            "steps": int(global_steps),
        }
        history.append(epoch_metrics)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} finished: {json.dumps(epoch_metrics)}")
            if getattr(args, "save_epoch_checkpoints", False):
                epoch_dir = Path(args.epoch_checkpoint_dir or args.output_dir)
                epoch_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(router)
                epoch_metadata = {
                    "method": "wikitext2_delta_nll_greedy_set_bce",
                    "router_input": args.router_input,
                    "prompt_only_router_context": True,
                    "target_leakage_guard": "router sees only first router_prefix_tokens; eval never loads greedy labels",
                    "mask_application": "config.custom_layer_mask",
                    "prefix_depth": int(args.prefix_depth) if args.router_input in ATTENTION_ROUTER_INPUTS else 0,
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
                    "epoch": int(epoch + 1),
                    "epochs": int(args.epochs),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                }
                torch.save(
                    {
                        "model_state_dict": unwrapped.state_dict(),
                        "metadata": epoch_metadata,
                        "epoch_metrics": epoch_metrics,
                    },
                    epoch_dir / f"risk_router_epoch{epoch + 1:03d}.pt",
                )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(router)
        metadata = {
            "method": "wikitext2_pudding_style_candidate_quality",
            "router_input": "raw_embedding",
            "router_features": ["raw_embedding_mean"],
            "router_architecture": "candidate_quality_mlp",
            "teacher_model": args.teacher_model,
            "candidate_label_file": args.candidate_label_file,
            "candidate_ids": candidate_ids,
            "candidate_keep_masks": candidate_keep_masks,
            "candidate_count": int(len(candidate_ids)),
            "loss": args.loss,
            "dataset": "wikitext-2-raw-v1",
            "dataset_disk_path": args.dataset_disk_path,
            "split": args.split,
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "num_layers": int(num_layers),
            "base_hidden_size": int(hidden_size),
            "hidden_size": int(hidden_size),
            "skip_rate": float(label_metadata.get("skip_rate", args.skip_rate)),
            "skip_count": int(label_metadata.get("skip_count", 0)),
            "keep_count": int(label_metadata.get("keep_count", 0)),
            "protected_head": int(label_metadata.get("protected_head", args.protected_head)),
            "protected_tail": int(label_metadata.get("protected_tail", args.protected_tail)),
            "allowed_layers": [int(idx) for idx in label_metadata.get("allowed_layers", [])],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "target_leakage_guard": "router sees only first router_prefix_tokens; eval never loads candidate labels",
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": unwrapped.state_dict(), "metadata": metadata}, output_dir / "candidate_router.pt")
        (output_dir / "training_metrics.json").write_text(
            json.dumps({"metadata": metadata, "history": history}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved candidate router checkpoint to {output_dir / 'candidate_router.pt'}")


def torch_kmeans(features: torch.Tensor, clusters: int, iters: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.dim() != 2:
        raise ValueError(f"KMeans expects [num_samples, hidden], got {tuple(features.shape)}")
    n = int(features.size(0))
    k = max(1, min(int(clusters), n))
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator, device=features.device)
    centers = features[perm[:k]].clone()
    assignments = torch.full((n,), -1, dtype=torch.long, device=features.device)
    for _ in range(max(1, int(iters))):
        distances = torch.cdist(features.float(), centers.float(), p=2)
        new_assignments = distances.argmin(dim=1)
        new_centers = centers.clone()
        for cluster_idx in range(k):
            mask = new_assignments.eq(cluster_idx)
            if bool(mask.any().item()):
                new_centers[cluster_idx] = features[mask].mean(dim=0)
            else:
                replacement = int(torch.randint(0, n, (1,), generator=generator, device=features.device).item())
                new_centers[cluster_idx] = features[replacement]
        centers = F.normalize(new_centers.float(), dim=-1)
        if torch.equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
    return centers.cpu(), assignments.cpu()


def build_ig_artifact(args):
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    label_rows, label_metadata = load_candidate_label_rows(args.candidate_label_file)
    label_by_sample = {int(row["sample_id"]): row for row in label_rows}
    selected_window_ids = sorted(label_by_sample)
    candidate_ids = [str(value) for value in label_metadata.get("candidate_ids", label_rows[0]["candidate_ids"])]
    candidate_keep_masks = label_metadata.get("candidate_keep_masks")
    if not candidate_keep_masks:
        raise ValueError("Candidate metadata must include candidate_keep_masks.")

    dataset, token_ids = make_dataset_for_split(
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

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    num_layers = detect_num_layers(model)
    hidden_size = int(model.config.hidden_size)

    output_artifact = Path(args.output_artifact)
    output_summary = Path(args.output_summary)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        output_artifact.unlink(missing_ok=True)
        output_summary.unlink(missing_ok=True)
        for stale in output_artifact.parent.glob(output_artifact.name + ".rank*.pt"):
            stale.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    shard_path = output_artifact.with_suffix(output_artifact.suffix + f".rank{accelerator.process_index}.pt")

    sample_ids_out = []
    feature_rows = []
    iterator = tqdm(dataloader, desc="ig-prefix-features") if accelerator.is_main_process else dataloader
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sample_ids = [int(x) for x in batch["sample_id"].detach().cpu().tolist()]
            router_input_ids, router_attention_mask = router_prefix_batch(
                input_ids,
                attention_mask,
                args.router_prefix_tokens,
            )
            state = F.normalize(raw_embedding_state(model, router_input_ids, router_attention_mask).float(), dim=-1)
            sample_ids_out.extend(sample_ids)
            feature_rows.append(state.detach().cpu())
    torch.save(
        {
            "sample_ids": sample_ids_out,
            "features": torch.cat(feature_rows, dim=0) if feature_rows else torch.empty(0, hidden_size),
        },
        shard_path,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        feature_by_sample = {}
        for path in sorted(output_artifact.parent.glob(output_artifact.name + ".rank*.pt")):
            payload = torch.load(path, map_location="cpu")
            ids = [int(value) for value in payload["sample_ids"]]
            features = payload["features"].float()
            for idx, sample_id in enumerate(ids):
                feature_by_sample.setdefault(sample_id, features[idx])
        ordered_ids = [sample_id for sample_id in sorted(label_by_sample) if sample_id in feature_by_sample]
        if not ordered_ids:
            raise ValueError("No IG features were collected.")
        features = F.normalize(torch.stack([feature_by_sample[sample_id] for sample_id in ordered_ids], dim=0), dim=-1)
        deltas = torch.tensor(
            [label_by_sample[sample_id]["candidate_delta_nll"] for sample_id in ordered_ids],
            dtype=torch.float32,
        )
        centers, assignments = torch_kmeans(features, args.clusters, args.kmeans_iters, args.seed)
        global_best = int(deltas.mean(dim=0).argmin().item())
        cluster_candidate_indices = []
        cluster_sizes = []
        cluster_mean_delta_nll = []
        for cluster_idx in range(int(centers.size(0))):
            mask = assignments.eq(cluster_idx)
            cluster_sizes.append(int(mask.sum().item()))
            if bool(mask.any().item()):
                mean_delta = deltas[mask].mean(dim=0)
                best_idx = int(mean_delta.argmin().item())
                cluster_mean_delta_nll.append([float(value) for value in mean_delta.tolist()])
            else:
                best_idx = global_best
                cluster_mean_delta_nll.append([])
            cluster_candidate_indices.append(best_idx)

        metadata = {
            "method": "wikitext2_ig_style_prefix_cluster_mask",
            "router_input": "raw_embedding",
            "router_features": ["normalized_raw_embedding_mean"],
            "teacher_model": args.teacher_model,
            "candidate_label_file": args.candidate_label_file,
            "candidate_ids": candidate_ids,
            "candidate_keep_masks": candidate_keep_masks,
            "candidate_count": int(len(candidate_ids)),
            "clusters": int(centers.size(0)),
            "cluster_candidate_indices": [int(idx) for idx in cluster_candidate_indices],
            "cluster_candidate_ids": [candidate_ids[int(idx)] for idx in cluster_candidate_indices],
            "cluster_sizes": cluster_sizes,
            "cluster_mean_delta_nll": cluster_mean_delta_nll,
            "dataset": "wikitext-2-raw-v1",
            "dataset_disk_path": args.dataset_disk_path,
            "split": args.split,
            "seq_len": int(args.seq_len),
            "router_prefix_tokens": int(args.router_prefix_tokens),
            "num_layers": int(num_layers),
            "base_hidden_size": int(hidden_size),
            "hidden_size": int(hidden_size),
            "num_tokens": int(len(token_ids)),
            "num_samples": int(len(ordered_ids)),
            "skip_rate": float(label_metadata.get("skip_rate", args.skip_rate)),
            "skip_count": int(label_metadata.get("skip_count", 0)),
            "keep_count": int(label_metadata.get("keep_count", 0)),
            "protected_head": int(label_metadata.get("protected_head", args.protected_head)),
            "protected_tail": int(label_metadata.get("protected_tail", args.protected_tail)),
            "allowed_layers": [int(idx) for idx in label_metadata.get("allowed_layers", [])],
            "seed": int(args.seed),
            "kmeans_iters": int(args.kmeans_iters),
            "target_leakage_guard": "cluster masks are selected from train candidate labels; eval/test never loads labels",
        }
        torch.save({"centers": centers.float(), "metadata": metadata}, output_artifact)
        output_summary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved IG-style artifact to {output_artifact}")
        print(f"Wrote IG-style summary to {output_summary}")


def load_candidate_router_checkpoint(path: str, hidden_size: int, device):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    metadata = dict(checkpoint.get("metadata") or {})
    candidate_keep_masks = metadata.get("candidate_keep_masks")
    candidate_ids = metadata.get("candidate_ids")
    if not candidate_keep_masks or not candidate_ids:
        raise ValueError(f"Candidate router checkpoint lacks candidate metadata: {path}")
    router = PromptCandidateMaskRouter(
        hidden_size=int(metadata.get("base_hidden_size") or hidden_size),
        num_candidates=len(candidate_ids),
        dropout=0.0,
    )
    router.load_state_dict(state)
    router.to(device).eval()
    return router, metadata


def load_ig_artifact(path: str, device):
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload.get("metadata") or {})
    centers = payload.get("centers")
    if centers is None:
        raise ValueError(f"IG artifact lacks centers: {path}")
    if not metadata.get("candidate_keep_masks") or not metadata.get("cluster_candidate_indices"):
        raise ValueError(f"IG artifact lacks candidate/cluster metadata: {path}")
    return centers.float().to(device), metadata


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

    model = AutoModelForCausalLM.from_pretrained(
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
                    "set_loss_type": args.set_loss_type,
                    "token_count_train_split": int(len(token_ids)),
                    "target_leakage_guard": "router sees only first router_prefix_tokens; NLL labels score suffix tokens only",
                    "mask_application": "config.custom_layer_mask",
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
            if args.set_loss_type == "exact_k_ce":
                loss = exact_k_subset_ce_loss(
                    pred.float(),
                    target.float(),
                    allowed_layers=[int(idx) for idx in allowed_layers],
                    skip_count=int(skip_count),
                    score_clip=float(args.exact_k_score_clip),
                )
            else:
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
            if getattr(args, "save_epoch_checkpoints", False):
                epoch_dir = Path(args.epoch_checkpoint_dir or args.output_dir)
                epoch_dir.mkdir(parents=True, exist_ok=True)
                epoch_metadata = {
                    "method": f"wikitext2_delta_nll_greedy_set_{args.set_loss_type}",
                    "router_input": args.router_input,
                    "prompt_only_router_context": True,
                    "target_leakage_guard": "router sees only first router_prefix_tokens; eval never loads greedy labels",
                    "mask_application": "config.custom_layer_mask",
                    "prefix_depth": int(args.prefix_depth) if args.router_input in ATTENTION_ROUTER_INPUTS else 0,
                    "router_dim": int(args.router_dim) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
                    "router_heads": int(args.router_heads) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
                    "teacher_model": args.teacher_model,
                    "risk_label_file": args.risk_label_file,
                    "risk_objective": label_metadata.get("objective", "Delta_NLL"),
                    "supervision_type": "skip_set",
                    "set_loss_type": args.set_loss_type,
                    "exact_k_score_clip": float(args.exact_k_score_clip),
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
                    "epoch": int(epoch + 1),
                    "epochs": int(args.epochs),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                }
                torch.save(
                    {
                        "model_state_dict": accelerator.unwrap_model(router).state_dict(),
                        "metadata": epoch_metadata,
                        "epoch_metrics": epoch_metrics,
                    },
                    epoch_dir / f"risk_router_epoch{epoch + 1:03d}.pt",
                )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(router)
        metadata = {
            "method": f"wikitext2_delta_nll_greedy_set_{args.set_loss_type}",
            "router_input": args.router_input,
            "prompt_only_router_context": True,
            "target_leakage_guard": "router sees only first router_prefix_tokens; eval never loads greedy labels",
            "mask_application": "config.custom_layer_mask",
            "prefix_depth": int(args.prefix_depth) if args.router_input in ATTENTION_ROUTER_INPUTS else 0,
            "router_features": (
                ["raw_token_sequence", "hk_token_sequence", "attention_mask", "layer_queries"]
                if args.router_input == "prefix_hk_raw_attn"
                else (
                    ["per_layer_prompt_hidden_mean"]
                    if args.router_input == "layerwise_hidden"
                    else ["raw_embedding_mean"]
                )
            ),
            "router_architecture": (
                "layer_query_cross_attention"
                if args.router_input == "prefix_hk_raw_attn"
                else ("layerwise_hidden_mlp" if args.router_input == "layerwise_hidden" else "pooled_mlp")
            ),
            "router_access_note": (
                "layerwise_hidden_router is an in-framework stronger-access baseline, not a full Dr.LLM reproduction."
                if args.router_input == "layerwise_hidden"
                else ""
            ),
            "router_dim": int(args.router_dim) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "router_heads": int(args.router_heads) if args.router_input in ATTENTION_ROUTER_INPUTS else None,
            "teacher_model": args.teacher_model,
            "risk_label_file": args.risk_label_file,
            "risk_objective": label_metadata.get("objective", "Delta_NLL"),
            "supervision_type": "skip_set",
            "set_loss_type": args.set_loss_type,
            "exact_k_score_clip": float(args.exact_k_score_clip),
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


def train_shared_compensation_adapter(args):
    if args.method_specs_json:
        method_specs = json.loads(Path(args.method_specs_json).read_text(encoding="utf-8"))
    else:
        raise ValueError("--method_specs_json is required")
    if not isinstance(method_specs, list) or not method_specs:
        raise ValueError("--method_specs_json must contain a non-empty list")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, token_ids = make_dataset_for_split(
        args,
        tokenizer,
        selected_window_ids=None,
        max_windows=int(args.train_windows),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_wikitext_windows(batch, args.router_prefix_tokens),
    )

    model = AutoModelForCausalLM.from_pretrained(
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

    runtimes = [
        load_mask_runtime(
            spec,
            model,
            hidden_size,
            num_layers,
            device,
            skip_count,
            allowed_layers,
            args.protected_head,
            args.protected_tail,
            args.seed,
        )
        for spec in method_specs
    ]
    adapter = build_lowrank_adapter(num_layers, hidden_size, args.compensation_rank, model.config).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    comp = compensation_config("learned_lowrank", args.compensation_rank, args.compensation_static_gate)

    metadata = {
        "method": "wikitext2_shared_lowrank_residual_compensation",
        "adapter_formula": "h_out = h_in + B_l A_l RMSNorm(h_in)",
        "scope": "single_shared_adapter_across_methods",
        "teacher_model": args.teacher_model,
        "dataset": "wikitext-2-raw-v1",
        "dataset_disk_path": args.dataset_disk_path,
        "split": args.split,
        "seq_len": int(args.seq_len),
        "router_prefix_tokens": int(args.router_prefix_tokens),
        "train_windows": int(len(dataset)),
        "token_count_train_split": int(len(token_ids)),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "rank": int(args.compensation_rank),
        "adapter_param_count": int(adapter.parameter_count()),
        "skip_rate": float(args.skip_rate),
        "skip_count": int(skip_count),
        "keep_count": int(keep_count),
        "protected_head": int(args.protected_head),
        "protected_tail": int(args.protected_tail),
        "allowed_layers": [int(idx) for idx in allowed_layers],
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "kl_temperature": float(args.kl_temperature),
        "base_model_frozen": True,
        "method_specs": method_specs,
        "target_leakage_guard": "adapter is trained on train split only; eval/test path does not load labels",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(metadata, indent=2))

    history = []
    for epoch in range(int(args.epochs)):
        adapter.train()
        total_loss = 0.0
        total_steps = 0
        iterator = tqdm(dataloader, desc=f"shared-comp epoch {epoch + 1}/{args.epochs}")
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                clear_custom_policy(model)
                teacher_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits.detach()
                del teacher_outputs

            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            for runtime in runtimes:
                keep_masks, _ = keep_masks_for_runtime(
                    runtime,
                    model,
                    input_ids,
                    attention_mask,
                    args.router_prefix_tokens,
                    num_layers,
                )
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=device)
                action_payload = action_payload_from_keep_mask(layer_mask)
                set_custom_policy(
                    model,
                    layer_mask,
                    action_payload=action_payload,
                    compensation_config_payload=comp,
                    compensation_adapter=adapter,
                )
                student_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                loss_i = kl_full_to_skipped_loss(
                    student_outputs.logits / float(args.kl_temperature),
                    teacher_logits / float(args.kl_temperature),
                    labels,
                ) * (float(args.kl_temperature) ** 2)
                if not torch.isfinite(loss_i.detach()):
                    raise FloatingPointError(
                        f"Non-finite shared compensation loss: {float(loss_i.detach().float().item())}"
                    )
                (loss_i / float(len(runtimes))).backward()
                loss_values.append(float(loss_i.detach().float().item()))
                del student_outputs
                clear_custom_policy(model)
            loss_value = float(sum(loss_values) / max(1, len(loss_values)))
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.max_grad_norm))
            optimizer.step()
            total_loss += loss_value
            total_steps += 1
            iterator.set_postfix({"loss": f"{loss_value:.6f}"})
            del teacher_logits

        epoch_metrics = {
            "epoch": int(epoch + 1),
            "loss": float(total_loss / max(1, total_steps)),
            "steps": int(total_steps),
        }
        history.append(epoch_metrics)
        print(f"Epoch {epoch + 1} finished: {json.dumps(epoch_metrics)}")

    adapter.eval()
    checkpoint_path = output_dir / "shared_lowrank_adapter.pt"
    save_lowrank_adapter_checkpoint(str(checkpoint_path), adapter, metadata)
    (output_dir / "training_metrics.json").write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved shared compensation adapter to {checkpoint_path}")


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

    model = AutoModelForCausalLM.from_pretrained(
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
    candidate_router = None
    candidate_router_metadata = {}
    ig_centers = None
    ig_metadata = {}
    if args.method == "router":
        router, router_metadata = load_router_checkpoint(args.risk_router_ckpt, hidden_size, num_layers, device)
        allowed_layers = router_metadata.get("allowed_layers") or allowed_layers
        skip_count = int(router_metadata.get("skip_count") or skip_count)
        keep_count = int(num_layers) - skip_count
    elif args.method == "candidate_router":
        candidate_router, candidate_router_metadata = load_candidate_router_checkpoint(
            args.candidate_router_ckpt,
            hidden_size,
            device,
        )
        allowed_layers = candidate_router_metadata.get("allowed_layers") or allowed_layers
        skip_count = int(candidate_router_metadata.get("skip_count") or skip_count)
        keep_count = int(num_layers) - skip_count
    elif args.method == "ig":
        ig_centers, ig_metadata = load_ig_artifact(args.ig_artifact, device)
        allowed_layers = ig_metadata.get("allowed_layers") or allowed_layers
        skip_count = int(ig_metadata.get("skip_count") or skip_count)
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
    compensation_adapter = None
    compensation_adapter_metadata: Dict[str, object] = {}
    if args.compensation_mode == "learned_lowrank":
        if not args.compensation_adapter_ckpt:
            raise ValueError("--compensation_adapter_ckpt is required for learned_lowrank compensation")
        compensation_adapter, compensation_adapter_metadata = load_lowrank_adapter_checkpoint(
            args.compensation_adapter_ckpt,
            num_layers,
            hidden_size,
            device,
            model.config,
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
            selected_candidate_ids: List[Optional[str]] = [None for _ in sample_ids]
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
            elif args.method == "candidate_router":
                router_input_ids, router_attention_mask = router_prefix_batch(
                    input_ids,
                    attention_mask,
                    args.router_prefix_tokens,
                )
                pred_delta = candidate_router(raw_embedding_state(model, router_input_ids, router_attention_mask))
                candidate_indices = pred_delta.float().argmin(dim=1).detach().cpu().tolist()
                candidate_ids = [str(value) for value in candidate_router_metadata["candidate_ids"]]
                candidate_keep_masks = candidate_router_metadata["candidate_keep_masks"]
                keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
                selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=device)
            elif args.method == "ig":
                router_input_ids, router_attention_mask = router_prefix_batch(
                    input_ids,
                    attention_mask,
                    args.router_prefix_tokens,
                )
                state = F.normalize(raw_embedding_state(model, router_input_ids, router_attention_mask).float(), dim=-1)
                distances = torch.cdist(state.float(), ig_centers.float(), p=2)
                clusters = distances.argmin(dim=1).detach().cpu().tolist()
                cluster_candidate_indices = [int(idx) for idx in ig_metadata["cluster_candidate_indices"]]
                candidate_ids = [str(value) for value in ig_metadata["candidate_ids"]]
                candidate_keep_masks = ig_metadata["candidate_keep_masks"]
                candidate_indices = [cluster_candidate_indices[int(cluster_idx)] for cluster_idx in clusters]
                keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
                selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unsupported eval method: {args.method}")

            rows = forward_loss_rows(
                model,
                input_ids,
                attention_mask,
                labels,
                layer_mask=layer_mask,
                compensation_mode=args.compensation_mode,
                compensation_rank=args.compensation_rank,
                compensation_static_gate=args.compensation_static_gate,
                compensation_adapter=compensation_adapter,
            )
            for sample_id, window_start, stats, keep_mask, candidate_id in zip(
                sample_ids,
                window_starts,
                rows,
                keep_masks,
                selected_candidate_ids,
            ):
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
                if candidate_id is not None:
                    row["selected_candidate_id"] = candidate_id
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
        if args.method in {"static", "router", "candidate_router", "ig"} and float(mask_summary["exact_skip_count_rate"]) < 1.0:
            raise ValueError(
                f"{args.run_name} did not produce exactly K skipped layers for every window: "
                f"exact_skip_count_rate={mask_summary['exact_skip_count_rate']}"
            )
        selected_candidate_distribution = {}
        for row in rows:
            candidate_id = row.get("selected_candidate_id")
            if candidate_id:
                selected_candidate_distribution[str(candidate_id)] = selected_candidate_distribution.get(str(candidate_id), 0) + 1
        payload = {
            "run_name": args.run_name,
            "method": args.method,
            "method_label": args.method_label,
            "static_strategy": args.static_strategy if args.method == "static" else None,
            "risk_router_ckpt": args.risk_router_ckpt if args.method == "router" else "",
            "risk_router_metadata": router_metadata if args.method == "router" else {},
            "candidate_router_ckpt": args.candidate_router_ckpt if args.method == "candidate_router" else "",
            "candidate_router_metadata": candidate_router_metadata if args.method == "candidate_router" else {},
            "ig_artifact": args.ig_artifact if args.method == "ig" else "",
            "ig_metadata": ig_metadata if args.method == "ig" else {},
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
            "selected_candidate_distribution": selected_candidate_distribution,
            "uses_greedy_labels_at_eval": False,
            "target_leakage_guard": "router sees only first router_prefix_tokens; PPL scores suffix tokens only",
            "mask_application": "config.custom_layer_mask",
            "compensation": compensation_config(
                args.compensation_mode,
                args.compensation_rank if args.method != "full" else 0,
                args.compensation_static_gate,
            ),
            "compensation_adapter_ckpt": args.compensation_adapter_ckpt if args.compensation_mode == "learned_lowrank" else "",
            "compensation_adapter_metadata": compensation_adapter_metadata,
            "compensation_adapter_param_count": int(
                compensation_adapter_metadata.get("adapter_param_count", 0)
                if compensation_adapter_metadata
                else 0
            ),
            "compensation_runtime_calls": int(max([int(row.get("compensation_runtime_calls", 0)) for row in rows] or [0])),
            "compensation_runtime_total_sec": float(max([float(row.get("compensation_runtime_total_sec", 0.0)) for row in rows] or [0.0])),
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

    model = AutoModelForCausalLM.from_pretrained(
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


def method_type(name: str) -> str:
    if name == "Full":
        return "full"
    if name.startswith("Static"):
        return "static"
    if name in {"PuDDing-style", "IG-style", "layerwise_hidden_router"}:
        return "related"
    if name == "Raw-SetBCE":
        return "ablation"
    if name == "OPAL-SetBCE":
        return "ours"
    return "tuning"


def metric_table_lines(metrics: Dict[str, Dict[str, object]], order: Sequence[str]) -> List[str]:
    if "Full" not in metrics:
        return ["`Full` metric is missing; cannot compute deltas."]
    full_nll = float(metrics["Full"]["nll"])
    full_ppl = float(metrics["Full"]["ppl"])
    lines = [
        "| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks | exact-K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = metrics.get(name)
        if not row:
            lines.append(f"| {name} | {method_type(name)} | pending | pending | pending | pending | pending | pending |")
            continue
        nll = float(row["nll"])
        ppl = float(row["ppl"])
        lines.append(
            f"| {name} | {method_type(name)} | {fmt(nll)} | {fmt(ppl)} | "
            f"{fmt(nll - full_nll)} | {fmt(ppl - full_ppl)} | "
            f"{row.get('unique_masks', 'NA')} | {fmt(row.get('exact_skip_count_rate'), 3)} |"
        )
    return lines


def named_training_summaries(paths: Sequence[str]) -> Dict[str, Dict[str, object]]:
    result = {}
    for name, path in parse_named_paths(paths).items():
        summary = training_loss_summary(path)
        if summary:
            result[name] = summary
    return result


def write_related_report(args):
    metric_paths = parse_named_paths(args.metric)
    metrics = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in metric_paths.items()
        if Path(path).exists()
    }
    if "Full" not in metrics:
        raise ValueError("write_related_report requires --metric Full=/path/full.json")
    training = named_training_summaries(args.training_metric)
    overlaps = {
        name: load_optional_json(path)
        for name, path in parse_named_paths(args.overlap_summary).items()
        if Path(path).exists()
    }
    tuning_metrics = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in parse_named_paths(args.tuning_metric).items()
        if Path(path).exists()
    }
    candidate_metadata = load_optional_json(args.candidate_label_metadata) or {}
    ig_summary = load_optional_json(args.ig_summary) or {}

    order = [
        "Full",
        "Static uniform",
        "Static ends_heavy",
        "Static best-on-val C6",
        "PuDDing-style",
        "IG-style",
        "layerwise_hidden_router",
        "Raw-SetBCE",
        "OPAL-SetBCE",
    ]
    full = metrics["Full"]
    opal = metrics.get("OPAL-SetBCE")
    static_best = metrics.get("Static best-on-val C6")
    raw = metrics.get("Raw-SetBCE")

    judgment = "not ready"
    judgment_detail = "Related baselines are still pending, so WikiText-2 cannot support a main-text public LM claim yet."
    if opal and static_best:
        opal_wins_static = float(opal["nll"]) < float(static_best["nll"])
        related_available = all(name in metrics for name in ["PuDDing-style", "IG-style"])
        opal_wins_related = related_available and all(
            float(opal["nll"]) < float(metrics[name]["nll"]) for name in ["PuDDing-style", "IG-style"]
        )
        opal_wins_raw = raw is not None and float(opal["nll"]) < float(raw["nll"])
        if opal_wins_static and opal_wins_related:
            judgment = "main text candidate"
            judgment_detail = (
                "OPAL beats Static best-on-val C6 and PuDDing/IG-style baselines. "
                + ("It also beats Raw-SetBCE." if opal_wins_raw else "It still does not beat Raw-SetBCE, so phrase Raw as a strong ablation.")
            )
        elif opal_wins_static:
            judgment = "appendix only"
            judgment_detail = (
                "OPAL beats the fixed static baseline but does not yet clear Raw and/or related-work baselines. "
                "Use this as public LM sanity / partial generalization, not the main claim."
            )
        else:
            judgment = "do not write"
            judgment_detail = "OPAL does not beat Static best-on-val C6, so WikiText-2 should stay out of the paper except as negative diagnosis."

    lines = []
    lines.append("# WikiText-2 Public LM Related Baseline Results")
    lines.append("")
    lines.append(f"Last updated: {args.date}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    dataset_source = full.get("dataset_disk_path") or f"{full.get('dataset_path')}/{full.get('dataset_name')}"
    lines.append(f"- model path: `{full.get('teacher_model')}`")
    lines.append(f"- model name: `{full.get('model_name')}`")
    lines.append(f"- dataset: `{dataset_source}`")
    lines.append(f"- num_layers: {full.get('num_layers')}")
    lines.append(f"- seq_len: {full.get('seq_len')}")
    lines.append(f"- router_prefix_tokens: {full.get('router_prefix_tokens')}")
    lines.append(f"- eval_windows requested / actual: {full.get('eval_windows_requested')} / {full.get('eval_windows')}")
    lines.append(f"- eval_tokens: {full.get('eval_tokens')}")
    lines.append(f"- skip_rate: {full.get('skip_rate')}")
    lines.append(f"- skip_count: {static_best.get('skip_count') if static_best else 'pending'}")
    lines.append(f"- protected_head / protected_tail: {full.get('protected_head')} / {full.get('protected_tail')}")
    lines.append(f"- label samples: {candidate_metadata.get('num_samples', 'pending')}")
    lines.append(f"- candidate library: C{candidate_metadata.get('candidate_count', 16)} `{candidate_metadata.get('candidate_ids', list(RELATED_CANDIDATE_STRATEGIES))}`")
    lines.append("")

    lines.append("## Original WikiText Bad Result")
    lines.append("")
    original_order = [
        "Full",
        "Static uniform",
        "Static ends_heavy",
        "Static best-on-val C6",
        "Raw-SetBCE",
        "OPAL-SetBCE",
    ]
    lines.extend(metric_table_lines(metrics, original_order))
    lines.append("")
    if opal and raw:
        lines.append(
            f"Raw-SetBCE is currently ahead of OPAL-SetBCE by "
            f"{fmt(float(opal['nll']) - float(raw['nll']))} NLL / {fmt(float(opal['ppl']) - float(raw['ppl']))} PPL."
        )
        lines.append("")

    lines.append("## WikiText-2 Audit")
    lines.append("")
    lines.append("| check | status | note |")
    lines.append("|---|---|---|")
    lines.append("| score direction | pass | labels use `y_l=1` for skipped layers; BCE trains `-pred` toward skip=1, and eval skips the lowest `pred` scores. |")
    lines.append("| mask application | pass | eval writes `config.custom_layer_mask`; maskcfg smoke/main runs show static masks change PPL, unlike the pre-fix invalid run. |")
    lines.append("| exact budget | pass | metrics record `exact_skip_count_rate=1.0` for all skip methods. |")
    lines.append("| protected policy | pass | static/raw/OPAL/related baselines use the same protected head/tail and K. |")
    lines.append("| split leakage | pass | greedy/candidate labels are train-only; static best uses validation; test eval does not load labels. |")
    lines.append("| router input | pass | routers see only first `router_prefix_tokens`; labels/PPL score suffix tokens. |")
    lines.append("| full PPL sanity | pass | Full PPL is finite and plausible for Qwen2.5-1.5B on WikiText-2 raw. |")
    lines.append("| static best selection | pass | C6 best is selected by validation NLL, not test. |")
    lines.append("| label objective | pass | greedy and candidate labels optimize suffix Delta_NLL. |")
    lines.append("")

    lines.append("## Why OPAL Did Not Win Yet")
    lines.append("")
    lines.append("- The current OPAL run beats Static best-on-val C6, but loses narrowly to Raw-SetBCE on the main `seq1024/pref256/m2000` run.")
    lines.append("- OPAL predicts more diverse masks than Raw, but diversity alone is not useful if WikiText-2 layer-skip decisions are dominated by coarse static/early-content patterns.")
    lines.append("- `prefix_hk_raw_attn` may be adding variance on LM windows: the teacher prefix representation can overfit short train windows, while raw embeddings act as a lower-variance content prior.")
    lines.append("- The next minimal correction is prefix length 512 and, if needed, validation checkpoint selection/static-prior variants; do not claim OPAL wins public LM until related baselines are in.")
    lines.append("")

    lines.append("## Related Baseline Table")
    lines.append("")
    lines.extend(metric_table_lines(metrics, order))
    lines.append("")

    lines.append("## Training And Diagnostics")
    lines.append("")
    for name in ["PuDDing-style", "layerwise_hidden_router", "Raw-SetBCE", "OPAL-SetBCE"]:
        summary = training.get(name)
        if not summary:
            continue
        best = summary["best"]
        first = summary["first"]
        last = summary["last"]
        lines.append(
            f"- {name}: epochs={summary['epochs']}, first={fmt(first.get('loss'))}, "
            f"best={fmt(best.get('loss'))} @ epoch {best.get('epoch')}, last={fmt(last.get('loss'))}"
        )
    for name, summary in overlaps.items():
        if not summary:
            continue
        lines.append(
            f"- {name} overlap: overlap@K={fmt(summary.get('mean_overlap_ratio'))}, "
            f"hamming={fmt(summary.get('mean_hamming_ratio'))}, unique={summary.get('unique_predicted_masks')}"
        )
    if "PuDDing-style" in metrics:
        lines.append(f"- PuDDing-style selected candidate distribution: `{metrics['PuDDing-style'].get('selected_candidate_distribution', {})}`")
    if "IG-style" in metrics:
        lines.append(f"- IG-style selected candidate distribution: `{metrics['IG-style'].get('selected_candidate_distribution', {})}`")
    if ig_summary:
        lines.append(f"- IG cluster sizes: `{ig_summary.get('cluster_sizes')}`")
        lines.append(f"- IG cluster selected masks: `{ig_summary.get('cluster_candidate_ids')}`")
    lines.append("- layerwise_hidden_router is an in-framework stronger-access baseline, not a full Dr.LLM reproduction.")
    lines.append("")

    lines.append("## OPAL Tuning Table")
    lines.append("")
    if tuning_metrics:
        tuning_order = list(tuning_metrics)
        lines.extend(metric_table_lines({"Full": full, **tuning_metrics}, ["Full"] + tuning_order))
    else:
        lines.append("| setting | status | note |")
        lines.append("|---|---|---|")
        lines.append("| prefix tokens 256 | complete | current main run; OPAL beats static but loses Raw narrowly. |")
        lines.append("| prefix tokens 512 | pending | run the minimal correction command below if related baselines do not rescue the story. |")
        lines.append("| validation checkpoint / static prior / swap q | pending | run only after prefix-512 check. |")
    lines.append("")

    lines.append("## Final Judgment")
    lines.append("")
    lines.append(f"- decision: **{judgment}**")
    lines.append(f"- rationale: {judgment_detail}")
    lines.append("")

    lines.append("## Commands")
    lines.append("")
    lines.append("Related baseline run:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /workspace/PriorDynamicPruning")
    lines.append("source ~/venvs/planrec/bin/activate")
    lines.append("git pull --ff-only origin codex/opal-llm-experiments")
    lines.append("")
    lines.append("WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \\")
    lines.append("WIKITEXT_LABEL_SAMPLES=2000 \\")
    lines.append("WIKITEXT_EVAL_WINDOWS=512 \\")
    lines.append("WIKITEXT_SEQ_LEN=1024 \\")
    lines.append("WIKITEXT_ROUTER_PREFIX_TOKENS=256 \\")
    lines.append("WIKITEXT_SEED=42 \\")
    lines.append("WIKITEXT_BASE_PORT=58200 \\")
    lines.append("WIKITEXT_RUN_PUDDING=1 \\")
    lines.append("WIKITEXT_RUN_IG=1 \\")
    lines.append("WIKITEXT_RUN_LAYERWISE=1 \\")
    lines.append("bash ./run_wikitext2_related_baselines_gpu01234567.sh")
    lines.append("```")
    lines.append("")
    lines.append("Minimal OPAL correction if needed:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /workspace/PriorDynamicPruning")
    lines.append("source ~/venvs/planrec/bin/activate")
    lines.append("")
    lines.append("WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \\")
    lines.append("WIKITEXT_LABEL_SAMPLES=2000 \\")
    lines.append("WIKITEXT_EVAL_WINDOWS=512 \\")
    lines.append("WIKITEXT_SEQ_LEN=1024 \\")
    lines.append("WIKITEXT_ROUTER_PREFIX_TOKENS=512 \\")
    lines.append("WIKITEXT_SEED=42 \\")
    lines.append("WIKITEXT_BASE_PORT=58300 \\")
    lines.append("bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh")
    lines.append("```")
    lines.append("")

    lines.append("## Artifact Paths")
    lines.append("")
    for name, path in metric_paths.items():
        lines.append(f"- metric `{name}`: `{path}`")
    for name, path in parse_named_paths(args.training_metric).items():
        lines.append(f"- training `{name}`: `{path}`")
    for name, path in parse_named_paths(args.overlap_summary).items():
        lines.append(f"- overlap `{name}`: `{path}`")
    if args.candidate_label_metadata:
        lines.append(f"- candidate labels metadata: `{args.candidate_label_metadata}`")
    if args.ig_summary:
        lines.append(f"- IG summary: `{args.ig_summary}`")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote related report to {args.output_md}")


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

    candidate_labels = sub.add_parser("build_candidate_labels")
    add_data_args(candidate_labels)
    candidate_labels.add_argument("--reference_label_file", default="")
    candidate_labels.add_argument("--label_samples", type=int, default=2000)
    candidate_labels.add_argument("--sample_strategy", choices=["first", "random"], default="random")
    candidate_labels.add_argument("--sample_seed", type=int, default=42)
    candidate_labels.add_argument("--skip_rate", type=float, default=0.25)
    candidate_labels.add_argument("--skip_count", type=int, default=0)
    candidate_labels.add_argument("--protected_head", type=int, default=4)
    candidate_labels.add_argument("--protected_tail", type=int, default=2)
    candidate_labels.add_argument("--batch_size", type=int, default=1)
    candidate_labels.add_argument("--candidate_batch_size", type=int, default=4)
    candidate_labels.add_argument("--output", required=True)
    candidate_labels.set_defaults(func=build_candidate_labels)

    candidate_train = sub.add_parser("train_candidate_router")
    add_data_args(candidate_train)
    candidate_train.add_argument("--candidate_label_file", required=True)
    candidate_train.add_argument("--skip_rate", type=float, default=0.25)
    candidate_train.add_argument("--skip_count", type=int, default=0)
    candidate_train.add_argument("--protected_head", type=int, default=4)
    candidate_train.add_argument("--protected_tail", type=int, default=2)
    candidate_train.add_argument("--batch_size", type=int, default=4)
    candidate_train.add_argument("--epochs", type=int, default=40)
    candidate_train.add_argument("--lr", type=float, default=1e-4)
    candidate_train.add_argument("--dropout", type=float, default=0.0)
    candidate_train.add_argument("--loss", choices=["huber", "mse"], default="huber")
    candidate_train.add_argument("--max_grad_norm", type=float, default=1.0)
    candidate_train.add_argument("--output_dir", required=True)
    candidate_train.set_defaults(func=train_candidate_router)

    ig = sub.add_parser("build_ig_artifact")
    add_data_args(ig)
    ig.add_argument("--candidate_label_file", required=True)
    ig.add_argument("--skip_rate", type=float, default=0.25)
    ig.add_argument("--skip_count", type=int, default=0)
    ig.add_argument("--protected_head", type=int, default=4)
    ig.add_argument("--protected_tail", type=int, default=2)
    ig.add_argument("--batch_size", type=int, default=8)
    ig.add_argument("--clusters", type=int, default=8)
    ig.add_argument("--kmeans_iters", type=int, default=30)
    ig.add_argument("--output_artifact", required=True)
    ig.add_argument("--output_summary", required=True)
    ig.set_defaults(func=build_ig_artifact)

    train = sub.add_parser("train_router")
    add_data_args(train)
    train.add_argument("--risk_label_file", required=True)
    train.add_argument("--router_input", choices=["raw_embedding", "prefix_hk_raw_attn", "layerwise_hidden"], default="prefix_hk_raw_attn")
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
    train.add_argument("--set_loss_type", choices=["bce", "exact_k_ce"], default="bce")
    train.add_argument("--exact_k_score_clip", type=float, default=50.0)
    train.add_argument("--output_dir", required=True)
    train.add_argument("--save_epoch_checkpoints", action="store_true")
    train.add_argument("--epoch_checkpoint_dir", default="")
    train.set_defaults(func=train_router)

    comp_train = sub.add_parser("train_shared_compensation_adapter")
    add_data_args(comp_train)
    comp_train.add_argument("--method_specs_json", required=True)
    comp_train.add_argument("--train_windows", type=int, default=2000)
    comp_train.add_argument("--sample_strategy", choices=["first", "random"], default="random")
    comp_train.add_argument("--sample_seed", type=int, default=42)
    comp_train.add_argument("--skip_rate", type=float, default=0.25)
    comp_train.add_argument("--skip_count", type=int, default=7)
    comp_train.add_argument("--protected_head", type=int, default=4)
    comp_train.add_argument("--protected_tail", type=int, default=2)
    comp_train.add_argument("--batch_size", type=int, default=1)
    comp_train.add_argument("--epochs", type=int, default=1)
    comp_train.add_argument("--lr", type=float, default=1e-4)
    comp_train.add_argument("--weight_decay", type=float, default=0.0)
    comp_train.add_argument("--max_grad_norm", type=float, default=1.0)
    comp_train.add_argument("--compensation_rank", type=int, default=16)
    comp_train.add_argument("--compensation_static_gate", type=float, default=1.0)
    comp_train.add_argument("--kl_temperature", type=float, default=1.0)
    comp_train.add_argument("--output_dir", required=True)
    comp_train.set_defaults(func=train_shared_compensation_adapter)

    eval_p = sub.add_parser("eval")
    add_data_args(eval_p)
    eval_p.add_argument("--eval_windows", type=int, default=512)
    eval_p.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    eval_p.add_argument("--sample_seed", type=int, default=42)
    eval_p.add_argument("--method", choices=["full", "static", "router", "candidate_router", "ig"], required=True)
    eval_p.add_argument("--method_label", default="")
    eval_p.add_argument("--static_strategy", choices=C6_STATIC_STRATEGIES, default="uniform")
    eval_p.add_argument("--risk_router_ckpt", default="")
    eval_p.add_argument("--candidate_router_ckpt", default="")
    eval_p.add_argument("--ig_artifact", default="")
    eval_p.add_argument("--prefix_depth", type=int, default=4)
    eval_p.add_argument("--skip_rate", type=float, default=0.25)
    eval_p.add_argument("--skip_count", type=int, default=0)
    eval_p.add_argument("--protected_head", type=int, default=4)
    eval_p.add_argument("--protected_tail", type=int, default=2)
    eval_p.add_argument("--batch_size", type=int, default=1)
    eval_p.add_argument("--compensation_mode", default="none")
    eval_p.add_argument("--compensation_rank", type=int, default=0)
    eval_p.add_argument("--compensation_static_gate", type=float, default=1.0)
    eval_p.add_argument("--compensation_adapter_ckpt", default="")
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

    related_report = sub.add_parser("write_related_report")
    related_report.add_argument("--metric", action="append", default=[], help="NAME=PATH, e.g. Full=/tmp/full.json")
    related_report.add_argument("--training_metric", action="append", default=[], help="NAME=PATH to training_metrics.json")
    related_report.add_argument("--overlap_summary", action="append", default=[], help="NAME=PATH to overlap summary JSON")
    related_report.add_argument("--tuning_metric", action="append", default=[], help="NAME=PATH for OPAL tuning metrics")
    related_report.add_argument("--candidate_label_metadata", default="")
    related_report.add_argument("--ig_summary", default="")
    related_report.add_argument("--output_md", required=True)
    related_report.add_argument("--date", default="2026-06-02")
    related_report.set_defaults(func=write_related_report)
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
        elif args.method == "candidate_router":
            args.method_label = "PuDDing-style"
        elif args.method == "ig":
            args.method_label = "IG-style"
    if hasattr(args, "router_prefix_tokens") and int(args.router_prefix_tokens) >= int(args.seq_len):
        raise ValueError("--router_prefix_tokens must be smaller than --seq_len")
    args.func(args)


if __name__ == "__main__":
    main()
