#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)
sys.path.insert(0, current_dir)

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import Collator
from lm_eval.models.utils_hf import pad_and_concat

from eval_wikitext_opal_ppl import (
    load_candidate_router_checkpoint,
    load_ig_artifact,
    load_router_checkpoint,
    raw_embedding_state,
    router_forward,
)
from wikitext_opal_utils import (
    allowed_layers_from_policy,
    clear_custom_policy,
    keep_masks_from_skip_risk,
    mask_key,
    resolve_skip_budget,
    set_custom_policy,
    skipped_layers_from_keep_mask,
    static_keep_mask,
    summarize_keep_masks,
    write_json,
)


TASKS = ("piqa", "openbookqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge")
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
METHOD_SLUGS = (
    "full",
    "static_ends_heavy",
    "static_best_on_val",
    "pudding",
    "ig",
    "layerwise",
    "raw_best_val",
    "opal_best_val",
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def finite_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


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


def pad_token_rows(rows: Sequence[Sequence[int]], pad_value: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(1, max(len(row) for row in rows))
    ids = torch.full((len(rows), max_len), int(pad_value), dtype=torch.long, device=device)
    attn = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for idx, row in enumerate(rows):
        values = list(row)
        ids[idx, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        attn[idx, : len(values)] = 1
    return ids, attn


class OpalHarnessLM(HFLM):
    """lm-evaluation-harness HFLM wrapper with OPAL layer-mask selection."""

    def __init__(
        self,
        method: str,
        method_label: str,
        static_strategy: str = "ends_heavy",
        risk_router_ckpt: str = "",
        candidate_router_ckpt: str = "",
        ig_artifact: str = "",
        skip_rate: float = 0.25,
        skip_count: int = 7,
        protected_head: int = 4,
        protected_tail: int = 2,
        router_prefix_tokens: int = 256,
        prefix_depth: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opal_method = str(method)
        self.opal_method_label = str(method_label)
        self.static_strategy = str(static_strategy)
        self.router_prefix_tokens = int(router_prefix_tokens)
        self.prefix_depth = int(prefix_depth)
        self.skip_count, self.keep_count = resolve_skip_budget(
            len(self.model.model.layers),
            float(skip_rate),
            int(skip_count),
        )
        self.allowed_layers = allowed_layers_from_policy(
            len(self.model.model.layers),
            int(protected_head),
            int(protected_tail),
        )
        self.mask_records: Dict[str, Dict[str, object]] = {}
        self.selected_candidate_distribution: Dict[str, int] = {}
        self.method_state: Dict[str, object] = {}
        hidden_size = int(self.model.config.hidden_size)
        num_layers = len(self.model.model.layers)

        if self.opal_method == "static":
            self.method_state["static_mask"] = static_keep_mask(
                self.static_strategy,
                num_layers=num_layers,
                skip_count=self.skip_count,
                protected_head=protected_head,
                protected_tail=protected_tail,
                seed=42,
            )
        elif self.opal_method == "router":
            router, metadata = load_router_checkpoint(risk_router_ckpt, hidden_size, num_layers, self.device)
            self.method_state["router"] = router
            self.method_state["router_metadata"] = metadata
            self.allowed_layers = metadata.get("allowed_layers") or self.allowed_layers
            self.skip_count = int(metadata.get("skip_count") or self.skip_count)
            self.keep_count = num_layers - self.skip_count
        elif self.opal_method == "candidate_router":
            router, metadata = load_candidate_router_checkpoint(candidate_router_ckpt, hidden_size, self.device)
            self.method_state["candidate_router"] = router
            self.method_state["candidate_router_metadata"] = metadata
            self.allowed_layers = metadata.get("allowed_layers") or self.allowed_layers
            self.skip_count = int(metadata.get("skip_count") or self.skip_count)
            self.keep_count = num_layers - self.skip_count
        elif self.opal_method == "ig":
            centers, metadata = load_ig_artifact(ig_artifact, self.device)
            self.method_state["ig_centers"] = centers
            self.method_state["ig_metadata"] = metadata
            self.allowed_layers = metadata.get("allowed_layers") or self.allowed_layers
            self.skip_count = int(metadata.get("skip_count") or self.skip_count)
            self.keep_count = num_layers - self.skip_count
        elif self.opal_method != "full":
            raise ValueError(f"Unsupported OPAL harness method: {self.opal_method}")

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        new_reqs = []
        for req in tqdm(
            requests,
            desc="Tokenizing inputs",
            disable=disable_tqdm,
        ):
            context, continuation = req.args
            task_name = getattr(req, "task_name", None) or "unknown"
            if context == "":
                continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
                context_enc, continuation_enc = (
                    ([self.prefix_token_id], continuation_enc)
                    if self.prefix_token_id != continuation_enc[0]
                    else (continuation_enc[:1], continuation_enc[1:])
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation, task_name), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _record_mask(self, task_name: str, context: str, keep_mask: Sequence[int], candidate_id: Optional[str] = None):
        key = json.dumps([str(task_name or "unknown"), context], ensure_ascii=False)
        if key not in self.mask_records:
            row = {
                "task": str(task_name or "unknown"),
                "mask_key": mask_key(keep_mask),
                "keep_mask": [int(v) for v in keep_mask],
                "skipped_layers": skipped_layers_from_keep_mask(keep_mask),
            }
            if candidate_id is not None:
                row["selected_candidate_id"] = str(candidate_id)
                self.selected_candidate_distribution[str(candidate_id)] = (
                    self.selected_candidate_distribution.get(str(candidate_id), 0) + 1
                )
            self.mask_records[key] = row

    def _mask_summary_by_task(self) -> Dict[str, Dict[str, object]]:
        by_task: Dict[str, List[List[int]]] = defaultdict(list)
        for row in self.mask_records.values():
            by_task[str(row["task"])].append([int(v) for v in row["keep_mask"]])
        expected = 0 if self.opal_method == "full" else self.skip_count
        return {task: summarize_keep_masks(masks, expected_skip_count=expected) for task, masks in by_task.items()}

    def _keep_masks_for_contexts(self, task_names: Sequence[str], contexts: Sequence[str], context_tokens: Sequence[Sequence[int]]):
        num_layers = len(self.model.model.layers)
        if self.opal_method == "full":
            keep_masks = [[1] * num_layers for _ in context_tokens]
            for task_name, context, keep_mask in zip(task_names, contexts, keep_masks):
                self._record_mask(task_name, context, keep_mask)
            return keep_masks
        if self.opal_method == "static":
            keep_masks = [list(self.method_state["static_mask"]) for _ in context_tokens]
            for task_name, context, keep_mask in zip(task_names, contexts, keep_masks):
                self._record_mask(task_name, context, keep_mask)
            return keep_masks

        prefix_rows = []
        for tokens in context_tokens:
            row = list(tokens[: max(1, int(self.router_prefix_tokens))])
            if not row:
                row = [self.prefix_token_id]
            prefix_rows.append(row)
        router_input_ids, router_attention_mask = pad_token_rows(prefix_rows, self.tokenizer.pad_token_id, self.device)

        selected_candidate_ids: List[Optional[str]] = [None for _ in prefix_rows]
        if self.opal_method == "router":
            metadata = self.method_state["router_metadata"]
            pred_risk = router_forward(
                self.method_state["router"],
                str(metadata.get("router_input") or "raw_embedding"),
                self.model,
                router_input_ids,
                router_attention_mask,
                num_layers,
                int(metadata.get("prefix_depth") or self.prefix_depth),
            )
            keep_masks = keep_masks_from_skip_risk(pred_risk, skip_count=self.skip_count, allowed_layers=self.allowed_layers)
        elif self.opal_method == "candidate_router":
            metadata = self.method_state["candidate_router_metadata"]
            pred_delta = self.method_state["candidate_router"](
                raw_embedding_state(self.model, router_input_ids, router_attention_mask)
            )
            candidate_indices = pred_delta.float().argmin(dim=1).detach().cpu().tolist()
            candidate_ids = [str(value) for value in metadata["candidate_ids"]]
            candidate_keep_masks = metadata["candidate_keep_masks"]
            keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
            selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        elif self.opal_method == "ig":
            metadata = self.method_state["ig_metadata"]
            state = F.normalize(raw_embedding_state(self.model, router_input_ids, router_attention_mask).float(), dim=-1)
            distances = torch.cdist(state.float(), self.method_state["ig_centers"].float(), p=2)
            clusters = distances.argmin(dim=1).detach().cpu().tolist()
            cluster_candidate_indices = [int(idx) for idx in metadata["cluster_candidate_indices"]]
            candidate_ids = [str(value) for value in metadata["candidate_ids"]]
            candidate_keep_masks = metadata["candidate_keep_masks"]
            candidate_indices = [cluster_candidate_indices[int(cluster_idx)] for cluster_idx in clusters]
            keep_masks = [[int(v) for v in candidate_keep_masks[int(idx)]] for idx in candidate_indices]
            selected_candidate_ids = [candidate_ids[int(idx)] for idx in candidate_indices]
        else:
            raise ValueError(f"Unsupported OPAL harness method: {self.opal_method}")

        for task_name, context, keep_mask, candidate_id in zip(task_names, contexts, keep_masks, selected_candidate_ids):
            self._record_mask(task_name, context, keep_mask, candidate_id)
        return keep_masks

    def _model_call_with_mask(self, inps: torch.Tensor, layer_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if layer_mask is None:
            clear_custom_policy(self.model)
        else:
            set_custom_policy(self.model, layer_mask)
        try:
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ):
                return self.model(inps).logits
        finally:
            clear_custom_policy(self.model)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req):
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts" if self.backend == "causal" and self.logits_cache else None,
            group_fn=_lookup_one_token_cont,
        )
        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size if self.batch_size != "auto" else override_bs if override_bs is not None else 0
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )
        if batch_fn is not None:
            self.batch_sizes = {}

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc=f"{self.opal_method_label} loglikelihood",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []
            task_names = []
            contexts = []
            context_token_rows = []
            padding_len_inp = None

            for request_str, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length
                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length + 1:
                    pass
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape
                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                task_names.append(request_str[2] if request_str and len(request_str) > 2 else "unknown")
                contexts.append(request_str[0] if request_str else "")
                context_token_rows.append(context_enc)

            assert padding_len_inp
            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
            keep_masks = self._keep_masks_for_contexts(task_names, contexts, context_token_rows)
            layer_mask = None
            if self.opal_method != "full":
                layer_mask = torch.tensor(keep_masks, dtype=torch.float32, device=self.device)
            multi_logits = F.log_softmax(
                self._model_call_with_mask(batched_inps, layer_mask),
                dim=-1,
                dtype=self.softmax_dtype,
            )

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list, strict=True
            ):
                contlen = len(cont_toks)
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len).unsqueeze(0)
                greedy_tokens = logits.argmax(dim=-1)
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
                    max_equal = (greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks).all()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                    answer = (float(logits.sum()), bool(max_equal))
                    res.append(answer)
                    if request_str is not None:
                        self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)


def metric_from_result(task_result: Dict[str, object], metric: str) -> Optional[float]:
    for key, value in task_result.items():
        if str(key).split(",", 1)[0] == metric:
            try:
                return float(value)
            except Exception:
                return None
    return None


def run_eval(args):
    tasks = parse_csv(args.tasks)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "start_eval",
                "run_name": args.run_name,
                "method": args.method,
                "method_label": args.method_label,
                "tasks": tasks,
                "limit": args.limit,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"[{args.run_name}] constructing OpalHarnessLM", flush=True)
    lm = OpalHarnessLM(
        pretrained=args.model,
        method=args.method,
        method_label=args.method_label,
        static_strategy=args.static_strategy,
        risk_router_ckpt=args.risk_router_ckpt,
        candidate_router_ckpt=args.candidate_router_ckpt,
        ig_artifact=args.ig_artifact,
        skip_rate=args.skip_rate,
        skip_count=args.skip_count,
        protected_head=args.protected_head,
        protected_tail=args.protected_tail,
        router_prefix_tokens=args.router_prefix_tokens,
        prefix_depth=args.prefix_depth,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=True,
        logits_cache=False,
    )
    print(f"[{args.run_name}] OpalHarnessLM ready; entering simple_evaluate", flush=True)
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit if args.limit > 0 else None,
        log_samples=False,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        bootstrap_iters=0,
    )
    print(f"[{args.run_name}] simple_evaluate complete; gathering mask records", flush=True)
    gathered_mask_records = lm.gather_object(lm.mask_records, dst=0)
    gathered_candidate_dists = lm.gather_object(lm.selected_candidate_distribution, dst=0)
    if lm.rank != 0 or results is None:
        return

    mask_records: Dict[str, Dict[str, object]] = {}
    for shard in gathered_mask_records:
        for key, value in (shard or {}).items():
            mask_records.setdefault(key, value)
    selected_candidate_distribution: Dict[str, int] = {}
    for shard in gathered_candidate_dists:
        for key, value in (shard or {}).items():
            selected_candidate_distribution[str(key)] = selected_candidate_distribution.get(str(key), 0) + int(value)

    by_task_masks: Dict[str, List[List[int]]] = defaultdict(list)
    for row in mask_records.values():
        by_task_masks[str(row["task"])].append([int(v) for v in row["keep_mask"]])
    expected_skip = 0 if args.method == "full" else lm.skip_count
    task_metrics = []
    for task in tasks:
        task_result = results.get("results", {}).get(task, {})
        acc = metric_from_result(task_result, "acc")
        acc_norm = metric_from_result(task_result, "acc_norm")
        masks = by_task_masks.get(task, [])
        mask_summary = summarize_keep_masks(masks, expected_skip_count=expected_skip) if masks else {}
        task_metrics.append(
            {
                "task": task,
                "acc": acc,
                "acc_norm": acc_norm,
                "num_examples": int((results.get("n-samples", {}).get(task) or {}).get("effective", 0)),
                "unique_masks": int(mask_summary.get("unique_masks", 0)),
                "exact_skip_count_rate": float(mask_summary.get("exact_skip_count_rate", 1.0 if args.method == "full" else 0.0)),
                "average_kept_layers": float(mask_summary.get("average_kept_layers", len(lm.model.model.layers) if args.method == "full" else 0.0)),
            }
        )
    payload = {
        "run_name": args.run_name,
        "seed": int(args.seed),
        "method": args.method,
        "method_label": args.method_label,
        "model": args.model,
        "tasks": tasks,
        "num_fewshot": int(args.num_fewshot),
        "max_length": int(args.max_length),
        "router_prefix_tokens": int(args.router_prefix_tokens),
        "skip_rate": float(args.skip_rate),
        "skip_count": int(0 if args.method == "full" else lm.skip_count),
        "protected_head": int(args.protected_head),
        "protected_tail": int(args.protected_tail),
        "task_metrics": task_metrics,
        "average_acc": finite_mean([row["acc"] for row in task_metrics if row["acc"] is not None]),
        "average_acc_norm": finite_mean([
            row["acc_norm"] if row["acc_norm"] is not None else row["acc"]
            for row in task_metrics
            if row["acc"] is not None
        ]),
        "unique_masks_mean": finite_mean([row["unique_masks"] for row in task_metrics]),
        "exact_skip_count_rate_mean": finite_mean([row["exact_skip_count_rate"] for row in task_metrics]),
        "average_kept_layers_mean": finite_mean([row["average_kept_layers"] for row in task_metrics]),
        "selected_candidate_distribution": selected_candidate_distribution,
        "lm_eval_task_results": results.get("results", {}),
        "lm_eval_n_samples": results.get("n-samples", {}),
        "lm_eval_versions": results.get("versions", {}),
        "mask_application": "config.custom_layer_mask",
        "compensation": "none",
        "target_leakage_guard": "lm-eval-harness builds prompts; router masks use context tokens only, never continuation tokens",
    }
    write_json(args.output_json, payload)
    print(f"[{args.run_name}] wrote {args.output_json}", flush=True)
    print(json.dumps({k: payload[k] for k in ("run_name", "method_label", "seed", "average_acc", "average_acc_norm", "unique_masks_mean")}, indent=2))


def load_json(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(args):
    output_root = Path(args.output_root)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    tasks = parse_csv(args.tasks)
    method_slugs = parse_csv(args.methods)
    rows = []
    payloads: Dict[Tuple[int, str], Dict[str, object]] = {}
    missing = []
    for seed in seeds:
        for slug in method_slugs:
            path = output_root / "metrics" / f"seed{seed}" / f"{slug}.json"
            payload = load_json(path)
            if payload is None:
                missing.append(str(path))
                continue
            payloads[(seed, str(payload["method_label"]))] = payload
            by_task = {row["task"]: row for row in payload.get("task_metrics", [])}
            for task in tasks:
                metric = by_task.get(task)
                if not metric:
                    missing.append(f"{path}:{task}")
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "method": str(payload["method_label"]),
                        "acc": metric.get("acc"),
                        "acc_norm": metric.get("acc_norm"),
                        "unique_masks": int(metric.get("unique_masks", 0)),
                        "exact_skip_count_rate": float(metric.get("exact_skip_count_rate", 0.0)),
                        "average_kept_layers": float(metric.get("average_kept_layers", 0.0)),
                    }
                )
    full_by_seed_task = {(row["seed"], row["task"]): row for row in rows if row["method"] == "Full"}
    for row in rows:
        full = full_by_seed_task.get((row["seed"], row["task"]))
        row["acc_norm_or_acc"] = row["acc_norm"] if row["acc_norm"] is not None else row["acc"]
        if full and full.get("acc"):
            row["retention_acc"] = float(row["acc"] / full["acc"]) if row["acc"] is not None else None
        else:
            row["retention_acc"] = None
        full_norm = full.get("acc_norm") if full else None
        full_norm = full_norm if full_norm is not None else (full.get("acc") if full else None)
        if full_norm and row["acc_norm_or_acc"] is not None:
            row["retention_acc_norm"] = float(row["acc_norm_or_acc"] / full_norm)
        else:
            row["retention_acc_norm"] = None

    lines = [
        "# LLM Eval Downstream Results",
        "",
        f"Last updated: {args.date}",
        "",
        "## Setup",
        "",
        f"- model: `{args.model}`",
        f"- evaluator: `lm-evaluation-harness` Python API (`simple_evaluate`)",
        f"- tasks: `{tasks}`",
        f"- seeds: `{seeds}`",
        f"- max_length: `{args.max_length}`",
        f"- router_prefix_tokens: `{args.router_prefix_tokens}`",
        f"- skip_rate / skip_count: `{args.skip_rate}` / `{args.skip_count}`",
        f"- protected_head / protected_tail: `{args.protected_head}` / `{args.protected_tail}`",
        "- compensation: `none`",
        "- mask leakage guard: harness builds context/continuation; router masks use context tokens only",
        "",
        "## Per-Seed Task Results",
        "",
        "| seed | task | method | acc | acc_norm | retention_acc | retention_acc_norm | unique_masks | exact_skip_count_rate | average_kept_layers |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for task in tasks:
            for method in METHOD_ORDER:
                row = next((r for r in rows if r["seed"] == seed and r["task"] == task and r["method"] == method), None)
                if row is None:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(seed),
                            task,
                            method,
                            fmt(row["acc"]),
                            fmt(row["acc_norm"]),
                            fmt(row["retention_acc"]),
                            fmt(row["retention_acc_norm"]),
                            str(row["unique_masks"]),
                            fmt(row["exact_skip_count_rate"], 3),
                            fmt(row["average_kept_layers"], 2),
                        ]
                    )
                    + " |"
                )

    lines += [
        "",
        "## Three-Seed Mean/Std",
        "",
        "| task | method | acc mean | acc std | acc_norm mean | acc_norm std | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean | average_kept_layers mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for method in METHOD_ORDER:
            vals = [row for row in rows if row["task"] == task and row["method"] == method]
            if not vals:
                continue
            norm_values = [row["acc_norm_or_acc"] for row in vals]
            lines.append(
                "| "
                + " | ".join(
                    [
                        task,
                        method,
                        fmt(finite_mean([row["acc"] for row in vals])),
                        fmt(finite_std([row["acc"] for row in vals])),
                        fmt(finite_mean(norm_values)),
                        fmt(finite_std(norm_values)),
                        fmt(finite_mean([row["retention_acc"] for row in vals])),
                        fmt(finite_mean([row["retention_acc_norm"] for row in vals])),
                        fmt(finite_mean([row["unique_masks"] for row in vals])),
                        fmt(finite_mean([row["exact_skip_count_rate"] for row in vals]), 3),
                        fmt(finite_mean([row["average_kept_layers"] for row in vals]), 2),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Six-Task Average",
        "",
        "| method | acc mean | acc std over seeds | acc_norm mean | acc_norm std over seeds | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_seed_norms: Dict[str, List[float]] = defaultdict(list)
    for method in METHOD_ORDER:
        seed_acc, seed_norm, seed_ret, seed_ret_norm, seed_unique, seed_exact = [], [], [], [], [], []
        for seed in seeds:
            vals = [row for row in rows if row["seed"] == seed and row["method"] == method]
            if not vals:
                continue
            seed_acc.append(finite_mean([row["acc"] for row in vals]))
            seed_norm.append(finite_mean([row["acc_norm_or_acc"] for row in vals]))
            seed_ret.append(finite_mean([row["retention_acc"] for row in vals]))
            seed_ret_norm.append(finite_mean([row["retention_acc_norm"] for row in vals]))
            seed_unique.append(finite_mean([row["unique_masks"] for row in vals]))
            seed_exact.append(finite_mean([row["exact_skip_count_rate"] for row in vals]))
        method_seed_norms[method] = seed_norm
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(finite_mean(seed_acc)),
                    fmt(finite_std(seed_acc)),
                    fmt(finite_mean(seed_norm)),
                    fmt(finite_std(seed_norm)),
                    fmt(finite_mean(seed_ret)),
                    fmt(finite_mean(seed_ret_norm)),
                    fmt(finite_mean(seed_unique)),
                    fmt(finite_mean(seed_exact), 3),
                ]
            )
            + " |"
        )

    opal = finite_mean(method_seed_norms["OPAL-SetBCE best-on-val"])
    lines += ["", "## Required Judgments", ""]
    for baseline in [
        "Raw-SetBCE best-on-val",
        "layerwise_hidden_router",
        "PuDDing-style",
        "IG-style",
        "Static best-on-val",
        "Static ends_heavy",
    ]:
        lines.append(f"- OPAL best-on-val beats `{baseline}` by six-task mean acc_norm-or-acc: `{opal > finite_mean(method_seed_norms[baseline])}`")
    opal_unique = [
        finite_mean([row["unique_masks"] for row in rows if row["seed"] == seed and row["method"] == "OPAL-SetBCE best-on-val"])
        for seed in seeds
    ]
    lines.append(f"- OPAL best-on-val unique_masks mean per seed: `{[round(v, 4) for v in opal_unique]}`")
    for method in ("PuDDing-style", "IG-style"):
        distributions = []
        for seed in seeds:
            payload = payloads.get((seed, method))
            if payload:
                distributions.append((seed, payload.get("selected_candidate_distribution", {})))
        collapsed = all(isinstance(dist, dict) and len(dist) == 1 and "ends_heavy" in dist for _, dist in distributions)
        lines.append(f"- {method} candidate distributions by seed: `{distributions}`")
        lines.append(f"- {method} degenerates to ends_heavy only: `{collapsed}`")
    if any(v <= 2 for v in opal_unique if math.isfinite(v)):
        lines.append("- Caveat: OPAL best-on-val unique_masks is very low on at least one seed; keep the static-like checkpoint caveat.")
    if missing:
        lines += ["", "## Missing Artifacts", ""]
        lines.extend(f"- `{item}`" for item in missing)

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="lm-evaluation-harness OPAL downstream runner")
    sub = parser.add_subparsers(dest="command", required=True)
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--tasks", default=",".join(TASKS))
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
    eval_p.add_argument("--router_prefix_tokens", type=int, default=256)
    eval_p.add_argument("--prefix_depth", type=int, default=4)
    eval_p.add_argument("--max_length", type=int, default=1024)
    eval_p.add_argument("--batch_size", default="8")
    eval_p.add_argument("--dtype", default="bfloat16")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--num_fewshot", type=int, default=0)
    eval_p.add_argument("--limit", type=float, default=0)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--run_name", required=True)
    eval_p.add_argument("--output_dir", default="")
    eval_p.add_argument("--output_json", required=True)
    eval_p.set_defaults(func=run_eval)

    report = sub.add_parser("summarize")
    report.add_argument("--output_root", required=True)
    report.add_argument("--output_md", required=True)
    report.add_argument("--model", default="/workspace/ckpts/Qwen2.5-1.5B")
    report.add_argument("--tasks", default=",".join(TASKS))
    report.add_argument("--methods", default=",".join(METHOD_SLUGS))
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
