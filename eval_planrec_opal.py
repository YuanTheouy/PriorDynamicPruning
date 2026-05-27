#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

from opal_llm.results import (
    append_summary_csv,
    build_payload,
    command_line,
    git_commit,
    load_ground_truths,
    load_valid_sids,
    result_dirs,
    summary_row,
    write_json,
)
from opal_llm.mask_library import (
    ACTION_COMPENSATE,
    ACTION_EXECUTE,
    ALLOWED_COMPENSATED_LAYERS,
    ALLOWED_COMPENSATION_RANKS,
    DEFAULT_MASK_STRATEGIES,
    LayerMaskSpec,
    build_action_plan,
    exact_topk_mask_from_scores,
    filter_masks,
    generate_mask,
    generate_mask_library,
    load_mask_library,
    local_threshold_mask_from_scores,
    mask_id_from_mask,
    prompt_feature_scores,
    save_mask_library,
    summarize_batch_masks,
    tensor_masks_to_lists,
)
from opal_llm.mask_utils import (
    actual_skip_rate,
    keep_count_from_skip_rate,
    mask_from_skip_risk,
    repair_structure_constraints,
)
from opal_llm.oracle_masks import load_oracle_cache, oracle_eval_fields
from opal_llm.quality_metrics import build_target_scoring_batch, quality_rows_from_logits
from opal_llm.timing import ComponentTimer, GenerationTimer

torch = None
Accelerator = None
DataLoader = None
AutoTokenizer = None
GenerationConfig = None
LogitsProcessorList = None
Qwen2ForCausalLM = None
ConstrainedLogitsProcessor = None
EvalSidDataset = None
OneLayerStudentModel = None
LayerRouter = None


CATEGORY_LABELS = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[index])
        row["index"] = index
        return row


def parse_int_csv(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_strategy_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def unpack_checkpoint(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = dict(payload.get("metadata") or {})
        for key in ("router_input_source", "prefix_depth", "top_k_layers", "num_layers"):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        return payload["model_state_dict"], metadata
    return payload, {}


def detect_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Could not detect decoder layer count")


def get_hash(x):
    return "-".join(str(_) for _ in x)


def build_prefix_allowed_tokens_fn(tokenizer, info_file: str, teacher_model: str):
    with open(info_file, "r", encoding="utf-8") as f:
        item_names = [line.split("\t")[0].strip() for line in f if line.strip()]

    info_semantic = [f"### Response:\n{item}\n" for item in item_names]
    if "llama" in teacher_model.lower():
        prefix_ids = [tokenizer(text).input_ids[1:] for text in info_semantic]
    else:
        prefix_ids = [tokenizer(text).input_ids for text in info_semantic]

    prefix_index = 4 if "gpt2" in teacher_model.lower() else 3
    hash_dict: Dict[str, set] = {}
    for ids in prefix_ids:
        ids.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ids)):
            key = get_hash(ids[:i] if i == prefix_index else ids[prefix_index:i])
            hash_dict.setdefault(key, set()).add(ids[i])

    allowed = {key: list(values) for key, values in hash_dict.items()}

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return allowed.get(get_hash(input_ids), [])

    return prefix_allowed_tokens_fn_semantic


def collate_left_pad(batch, pad_token_id: int):
    batch = [row for row in batch if row is not None]
    max_len = max(len(row["input_ids"]) for row in batch)
    padded_input_ids = []
    attention_masks = []
    indices = []

    for row in batch:
        ids = row["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        pad = max_len - len(ids)
        padded_input_ids.append(torch.tensor([pad_token_id] * pad + ids, dtype=torch.long))
        attention_masks.append(torch.tensor([0] * pad + [1] * len(ids), dtype=torch.long))
        indices.append(int(row["index"]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
        "index": torch.tensor(indices, dtype=torch.long),
    }


def load_or_build_masks(args, num_layers: int) -> List[LayerMaskSpec]:
    mask_library = args.mask_library or args.template_library
    save_mask_path = args.save_mask_library or args.save_template_library
    if mask_library:
        masks = load_mask_library(mask_library)
    else:
        budgets = parse_int_csv(args.budgets) if args.budgets else [args.top_k_layers]
        masks = generate_mask_library(
            num_layers=num_layers,
            budgets=budgets,
            strategies=parse_strategy_csv(args.mask_strategies),
            random_masks_per_budget=args.random_masks_per_budget,
            seed=args.seed,
        )
        if save_mask_path:
            save_mask_library(save_mask_path, masks)
    return masks


def static_mask_from_args(args, num_layers: int, mask_specs: Sequence[LayerMaskSpec]) -> LayerMaskSpec:
    selected_mask_id = args.mask_id or args.template_id
    if selected_mask_id:
        return filter_masks(mask_specs, mask_id=selected_mask_id)[0]
    if args.static_strategy:
        mask = generate_mask(args.static_strategy, num_layers, args.top_k_layers, seed=args.seed)
        return LayerMaskSpec(
            mask_id=f"{args.static_strategy}_k{sum(mask)}",
            strategy=args.static_strategy,
            budget=sum(mask),
            num_layers=num_layers,
            mask=mask,
            metadata={"source": "static_strategy"},
        )
    return filter_masks(mask_specs, budget=args.top_k_layers)[0]


def decode_generation(tokenizer, teacher_model: str, sequences, max_len: int) -> List[str]:
    completions = sequences[:, max_len:]
    if "llama" in teacher_model.lower():
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
    return [text.split("Response:\n")[-1].strip() for text in decoded]


def group_beam_outputs(decoded: Sequence[str], batch_size: int, num_beams: int) -> List[List[str]]:
    return [list(decoded[i * num_beams : (i + 1) * num_beams]) for i in range(batch_size)]


def rank_of(predictions: Sequence[str], target: str) -> int:
    target = str(target).strip(' \n"')
    for idx, pred in enumerate(predictions):
        if str(pred).strip(' \n"') == target:
            return idx
    return 1_000_000


def load_student_router(args, sid_token_ids: Sequence[int], num_layers: int, device):
    if not args.student_ckpt or not args.policy_ckpt:
        raise ValueError("--student_ckpt and --policy_ckpt are required for dynamic planner evaluation")

    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    student_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student_state, _ = unpack_checkpoint(student_ckpt)
    student.load_state_dict(student_state)
    student.to(dtype_from_precision(args.precision)).to(device).eval()

    hidden_size = student.backbone.config.hidden_size
    router = LayerRouter(hidden_size=hidden_size, num_layers=num_layers, top_k=args.top_k_layers)
    router_ckpt = torch.load(args.policy_ckpt, map_location="cpu")
    router_state, _ = unpack_checkpoint(router_ckpt)
    router.load_state_dict(router_state)
    router.to(dtype_from_precision(args.precision)).to(device).eval()
    return student, router


def load_policy_router(args, hidden_size: int, num_layers: int, device):
    if not args.policy_ckpt:
        if args.allow_fallback_router:
            return None, {"router_input_source": "deterministic_prefix_fallback"}
        raise ValueError("--policy_ckpt is required for --method opal. Use --allow_fallback_router only for debugging.")
    router = LayerRouter(hidden_size=hidden_size, num_layers=num_layers, top_k=args.top_k_layers)
    router_ckpt = torch.load(args.policy_ckpt, map_location="cpu")
    router_state, metadata = unpack_checkpoint(router_ckpt)
    router_input_source = str(metadata.get("router_input_source", "legacy_unknown"))
    valid_prefix_sources = {"teacher_prefix", "prefix_hidden", "opal_prefix_hidden"}
    if router_input_source not in valid_prefix_sources and not args.allow_legacy_policy_ckpt:
        raise ValueError(
            "OPAL requires a policy checkpoint trained on teacher prefix hidden states. "
            f"Checkpoint source is {router_input_source!r}. "
            "Use --allow_legacy_policy_ckpt only for ablations/debugging."
        )
    router.load_state_dict(router_state)
    router.to(dtype_from_precision(args.precision)).to(device).eval()
    return router, metadata


def pool_request_state(last_hidden_state, attention_mask):
    # MiniOneRec uses left padding for generation; the last active prompt token is
    # therefore the final column for every non-empty row.
    del attention_mask
    return last_hidden_state[:, -1, :]


def fallback_utility_scores(state, num_layers: int):
    rows = []
    state = state.detach().float()
    for row in state:
        summary = float(row.mean().item())
        energy = float(row.pow(2).mean().sqrt().item())
        scores = []
        for layer in range(num_layers):
            pos = (layer + 1) / max(1, num_layers)
            score = energy * (0.5 + pos) + 0.1 * summary * ((layer % 3) - 1)
            scores.append(score)
        rows.append(scores)
    return torch.tensor(rows, dtype=torch.float32, device=state.device)


def dynamic_masks(
    args,
    student,
    router,
    input_ids,
    attention_mask,
    num_layers: int,
):
    with torch.no_grad():
        student_out = student(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = student_out["last_hidden_state"]
        state = pool_request_state(last_hidden_state, attention_mask)
        mask, scores = router(state)

    hard_masks = tensor_masks_to_lists(mask)
    mask_ids = [mask_id_from_mask(row, prefix="student_router") for row in hard_masks]
    return hard_masks, mask_ids, scores.detach().float().cpu().tolist()


def layerwise_router_masks(args, student, router, input_ids, attention_mask, num_layers: int):
    if student is not None and router is not None:
        with torch.no_grad():
            student_out = student(input_ids=input_ids, attention_mask=attention_mask)
            state = pool_request_state(student_out["last_hidden_state"], attention_mask)
            _, scores = router(state)
    else:
        features = input_guided_features(input_ids, attention_mask, args.input_guided_num_bins)
        scores = prompt_feature_scores(features, num_layers, args.seed, args.input_guided_feature_set).to(input_ids.device)
    mask_tensor = local_threshold_mask_from_scores(
        scores,
        threshold=args.layerwise_threshold,
        top_k=args.top_k_layers,
        match_budget=args.layerwise_match_budget,
    )
    masks = tensor_masks_to_lists(mask_tensor)
    mask_ids = [mask_id_from_mask(row, prefix="layerwise") for row in masks]
    return masks, mask_ids, scores.detach().float().cpu().tolist()


def opal_prefix_masks(
    args,
    model,
    router,
    input_ids,
    attention_mask,
    num_layers: int,
    component_timer: ComponentTimer,
    is_warmup: bool = False,
):
    prefix_depth = max(0, min(int(args.prefix_depth), num_layers))
    prefix_mask = [1 if idx < prefix_depth else 0 for idx in range(num_layers)]

    def run_prefix():
        set_custom_policy(
            model,
            mask_payload=prefix_mask,
            action_payload=[ACTION_EXECUTE if keep else 0 for keep in prefix_mask],
            compensation_config={"mode": "none", "rank": 0},
        )
        try:
            with torch.no_grad():
                return model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
        finally:
            clear_custom_policy(model)

    prefix_hidden, prefix_latency = component_timer.measure(
        "prefix",
        run_prefix,
        samples=input_ids.size(0),
        metadata={"prefix_depth": prefix_depth, "warmup": bool(is_warmup)},
    )
    state = pool_request_state(prefix_hidden, attention_mask)

    def run_router():
        with torch.no_grad():
            if router is not None:
                _, scores = router(state)
                return scores.detach().float()
            return fallback_utility_scores(state, num_layers)

    scores, router_latency = component_timer.measure(
        "router",
        run_router,
        samples=input_ids.size(0),
        metadata={
            "router_source": "policy_ckpt" if router is not None else "deterministic_prefix_fallback",
            "warmup": bool(is_warmup),
        },
    )
    mask_tensor = exact_topk_mask_from_scores(
        scores,
        top_k=args.top_k_layers,
        prefix_depth=prefix_depth,
        tail_keep=args.tail_keep,
    )
    masks = tensor_masks_to_lists(mask_tensor)
    mask_ids = [mask_id_from_mask(row, prefix="opal") for row in masks]
    return masks, mask_ids, scores.cpu().tolist(), prefix_latency, router_latency


def prefix_hidden_state(
    args,
    model,
    input_ids,
    attention_mask,
    num_layers: int,
    component_timer: ComponentTimer,
    is_warmup: bool = False,
):
    prefix_depth = max(0, min(int(args.prefix_depth), num_layers))
    prefix_mask = [1 if idx < prefix_depth else 0 for idx in range(num_layers)]

    def run_prefix():
        set_custom_policy(
            model,
            mask_payload=prefix_mask,
            action_payload=[ACTION_EXECUTE if keep else 0 for keep in prefix_mask],
            compensation_config={"mode": "none", "rank": 0},
        )
        try:
            with torch.no_grad():
                return model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
        finally:
            clear_custom_policy(model)

    return component_timer.measure(
        "prefix",
        run_prefix,
        samples=input_ids.size(0),
        metadata={"prefix_depth": prefix_depth, "warmup": bool(is_warmup), "method": "opal_q"},
    )[0]


def opal_q_masks(
    args,
    model,
    router,
    oracle_cache: Dict[int, Dict[str, object]],
    input_ids,
    attention_mask,
    batch_indices: Sequence[int],
    num_layers: int,
    component_timer: ComponentTimer,
    is_warmup: bool = False,
):
    keep_count = keep_count_from_skip_rate(num_layers, args.skip_rate, args.top_k_layers)
    prefix_hidden = prefix_hidden_state(
        args,
        model,
        input_ids,
        attention_mask,
        num_layers,
        component_timer,
        is_warmup=is_warmup,
    )
    state = pool_request_state(prefix_hidden, attention_mask)

    def run_router():
        with torch.no_grad():
            if router is not None:
                _, scores = router(state)
                return scores.detach().float()
            return fallback_utility_scores(state, num_layers)

    keep_scores, _ = component_timer.measure(
        "router",
        run_router,
        samples=input_ids.size(0),
        metadata={
            "router_source": "policy_ckpt" if router is not None else "deterministic_prefix_fallback",
            "warmup": bool(is_warmup),
            "method": "opal_q",
            "opal_stage": int(args.opal_stage),
        },
    )
    skip_risks = keep_scores.detach().float().cpu().tolist()
    keep_scores_list = keep_scores.detach().float().cpu().tolist()

    masks = []
    mask_ids = []
    row_metadata = []
    for row_pos, (sample_index, row_risk) in enumerate(zip(batch_indices, skip_risks)):
        base_mask = mask_from_skip_risk(row_risk, keep_count=keep_count)
        structure_stats = {
            "max_consecutive_skips_actual": None,
            "stage_keep_counts": [],
            "structure_penalty_value": 0.0,
        }
        selected_mask = base_mask
        if int(args.opal_stage) >= 3 and float(args.structure_penalty) > 0:
            selected_mask, structure_stats = repair_structure_constraints(
                base_mask,
                row_risk,
                max_consecutive=args.max_consecutive_skips,
                num_stages=args.num_stages,
                min_keep_per_stage=args.min_keep_per_stage,
            )

        oracle_fields = {}
        if int(args.opal_stage) >= 2:
            oracle_fields = oracle_eval_fields(oracle_cache.get(int(sample_index)), selected_mask)

        masks.append(selected_mask)
        mask_ids.append(mask_id_from_mask(selected_mask, prefix=f"opal_q_s{int(args.opal_stage)}"))
        row_metadata.append(
            {
                "method": "opal_q",
                "opal_stage": int(args.opal_stage),
                "prefix_depth": int(args.prefix_depth),
                "skip_rate": actual_skip_rate(selected_mask),
                "requested_skip_rate": float(args.skip_rate) if float(args.skip_rate) >= 0 else None,
                "keep_score": keep_scores_list[row_pos],
                "skip_risk": row_risk,
                "kept_layer_count": sum(int(v) for v in selected_mask),
                **structure_stats,
                **oracle_fields,
            }
        )
    return masks, mask_ids, keep_scores_list, skip_risks, row_metadata


def build_actions_for_batch(args, masks: Sequence[Sequence[int]], scores: Optional[Sequence[Sequence[float]]] = None):
    if scores is None:
        scores = [[float(v) for v in mask] for mask in masks]
    action_plans = [
        build_action_plan(
            execution_mask=mask,
            scores=row_scores,
            compensation=args.compensation,
            max_compensated_skipped_layers=args.max_compensated_skipped_layers,
            margin_delta=args.compensation_margin_delta,
            margin_tau=args.compensation_margin_tau,
            static_gate=args.static_compensation_gate,
        )
        for mask, row_scores in zip(masks, scores)
    ]
    return action_plans


def compensation_config_from_args(args, action_plans: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "mode": args.compensation,
        "rank": int(args.comp_rank),
        "max_compensated_skipped_layers": int(args.max_compensated_skipped_layers),
        "static_gate": float(args.static_compensation_gate),
        "gates": [plan["compensation_gates"] for plan in action_plans],
    }


def timed_action_plans(
    args,
    masks: Sequence[Sequence[int]],
    scores: Optional[Sequence[Sequence[float]]],
    component_timer: ComponentTimer,
    metadata: Optional[Dict[str, object]] = None,
):
    timer_metadata = {"compensation": args.compensation, "comp_rank": args.comp_rank}
    timer_metadata.update(metadata or {})
    return component_timer.measure(
        "compensation",
        lambda: build_actions_for_batch(args, masks, scores),
        samples=len(masks),
        metadata=timer_metadata,
    )[0]


def input_guided_masks(args, input_ids, attention_mask, num_layers: int):
    """Select masks from prompt/input features, without executing an LLM prefix.

    This is a lightweight PuDDing/IG-style baseline hook. It intentionally uses
    only cheap prompt features, so OPAL gains can be attributed to internal prefix
    evidence rather than to a finite mask library.
    """
    features = input_guided_features(input_ids, attention_mask, args.input_guided_num_bins)
    calibrated_map = getattr(args, "_input_guided_calibrated_map", {}) or {}
    scores = prompt_feature_scores(features, num_layers, args.seed, args.input_guided_feature_set)
    if args.input_guided_selector == "length":
        scores = prompt_feature_scores(features, num_layers, args.seed, "length")
    elif args.input_guided_selector == "hash":
        scores = prompt_feature_scores(features, num_layers, args.seed, "hash")
    elif args.input_guided_selector == "calibrated" and calibrated_map:
        calibrated_masks = []
        for feature in features:
            key = input_guided_feature_key(feature, args.input_guided_feature_set)
            mask = calibrated_map.get(key)
            calibrated_masks.append(mask if mask is not None and len(mask) == num_layers else None)
        fallback_tensor = exact_topk_mask_from_scores(scores.to(input_ids.device), top_k=args.top_k_layers)
        fallback_masks = tensor_masks_to_lists(fallback_tensor)
        masks = []
        mask_ids = []
        for calibrated_mask, fallback_mask in zip(calibrated_masks, fallback_masks):
            if calibrated_mask is not None:
                row_mask = [int(v) for v in calibrated_mask]
                masks.append(row_mask)
                mask_ids.append(mask_id_from_mask(row_mask, prefix="input_calibrated"))
            else:
                masks.append(fallback_mask)
                mask_ids.append(mask_id_from_mask(fallback_mask, prefix="input_guided_fallback"))
        row_features = [summarize_input_guided_feature(feature, args.input_guided_feature_set) for feature in features]
        return masks, mask_ids, row_features, scores.tolist()

    mask_tensor = exact_topk_mask_from_scores(scores.to(input_ids.device), top_k=args.top_k_layers)
    masks = tensor_masks_to_lists(mask_tensor)
    mask_ids = [mask_id_from_mask(mask, prefix="input_guided") for mask in masks]
    row_features = [summarize_input_guided_feature(feature, args.input_guided_feature_set) for feature in features]
    return masks, mask_ids, row_features, scores.tolist()


def input_guided_features(input_ids, attention_mask, num_bins: int) -> List[Dict[str, int]]:
    num_bins = max(1, int(num_bins))
    lengths = attention_mask.detach().long().sum(dim=1).cpu().tolist()
    token_sums = (input_ids.detach().long() * attention_mask.detach().long()).sum(dim=1).cpu().tolist()
    last_tokens = []
    for row, mask in zip(input_ids.detach().long(), attention_mask.detach().long()):
        active = row[mask.bool()]
        last_tokens.append(int(active[-1].item()) if active.numel() else 0)

    features = []
    for length, token_sum, last_token in zip(lengths, token_sums, last_tokens):
        selector_value = int(token_sum) + 131 * int(length) + 17 * int(last_token)
        features.append(
            {
                "length": int(length),
                "length_bin": int(length) % num_bins,
                "token_sum": int(token_sum),
                "token_hash_bin": int(token_sum) % num_bins,
                "last_token_id": int(last_token),
                "last_token_bin": int(last_token) % num_bins,
                "selector_value": int(selector_value),
                "selector_value_bin": int(selector_value) % num_bins,
            }
        )
    return features


def input_guided_feature_key(feature: Dict[str, int], feature_set: str) -> str:
    if feature_set == "length":
        parts = [feature["length_bin"]]
    elif feature_set == "hash":
        parts = [feature["token_hash_bin"]]
    else:
        parts = [feature["length_bin"], feature["token_hash_bin"], feature["last_token_bin"]]
    return "|".join(str(int(part)) for part in parts)


def summarize_input_guided_feature(feature: Dict[str, int], feature_set: str) -> Dict[str, int]:
    return {
        "length": int(feature["length"]),
        "length_bin": int(feature["length_bin"]),
        "token_sum_mod": int(feature["token_sum"]) % 1_000_003,
        "token_hash_bin": int(feature["token_hash_bin"]),
        "last_token_id": int(feature["last_token_id"]),
        "last_token_bin": int(feature["last_token_bin"]),
        "selector_value_mod": int(feature["selector_value"]) % 1_000_003,
        "selector_value_bin": int(feature["selector_value_bin"]),
        "feature_key": input_guided_feature_key(feature, feature_set),
    }


def load_input_guided_calibration(args) -> Dict[str, List[int]]:
    if not args.input_guided_calibration_json:
        if args.input_guided_selector == "calibrated":
            raise ValueError("--input_guided_calibration_json is required for calibrated input-guided selection")
        return {}
    payload = json.loads(open(args.input_guided_calibration_json, encoding="utf-8").read())
    rows = payload.get("predictions", [])
    votes: Dict[str, Counter] = defaultdict(Counter)
    rank_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    rank_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    mask_by_id: Dict[str, List[int]] = {}
    for row in rows:
        feature = row.get("input_guided_features") or row.get("input_features") or {}
        if not feature:
            continue
        key = str(feature.get("feature_key") or input_guided_feature_key(feature, args.input_guided_feature_set))
        oracle_mask_id = str(row.get("mask_id") or row.get("template_id") or "")
        oracle_mask = row.get("execution_mask") or row.get("layer_mask") or []
        if oracle_mask_id and oracle_mask:
            mask_by_id[oracle_mask_id] = [int(v) for v in oracle_mask]
            votes[key][oracle_mask_id] += 1
        for candidate in row.get("oracle_candidates", []) or []:
            mask_id = str(candidate.get("mask_id") or candidate.get("template_id") or "")
            candidate_mask = candidate.get("execution_mask") or candidate.get("layer_mask") or []
            if not mask_id:
                continue
            if candidate_mask:
                mask_by_id[mask_id] = [int(v) for v in candidate_mask]
            rank_sums[(key, mask_id)] += float(candidate.get("rank", 1_000_000))
            rank_counts[(key, mask_id)] += 1

    calibrated = {}
    keys = set(votes)
    keys.update(key for key, _ in rank_sums)
    for key in keys:
        scored = [
            (rank_sums[(key, mask_id)] / rank_counts[(key, mask_id)], mask_id)
            for row_key, mask_id in rank_counts
            if row_key == key and rank_counts[(key, mask_id)] > 0
        ]
        if scored:
            calibrated[key] = mask_by_id.get(min(scored)[1])
        elif votes.get(key):
            calibrated[key] = mask_by_id.get(votes[key].most_common(1)[0][0])
    calibrated = {key: value for key, value in calibrated.items() if value}
    if args.input_guided_selector == "calibrated" and not calibrated:
        raise ValueError(f"No calibrated input-guided entries found in {args.input_guided_calibration_json}")
    return calibrated


def set_custom_policy(model, mask_payload, action_payload=None, compensation_config=None):
    model.config.custom_layer_mask = mask_payload
    model.config.custom_layer_actions = action_payload
    model.config.custom_compensation_config = compensation_config or {"mode": "none", "rank": 0}
    model.config.custom_compensation_runtime_stats = {"calls": 0, "total_sec": 0.0}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = mask_payload
                layer.self_attn.config.custom_layer_actions = action_payload
                layer.self_attn.config.custom_compensation_config = compensation_config or {"mode": "none", "rank": 0}
                layer.self_attn.config.custom_compensation_runtime_stats = model.config.custom_compensation_runtime_stats


def clear_custom_policy(model):
    stats = getattr(model.config, "custom_compensation_runtime_stats", {"calls": 0, "total_sec": 0.0})
    model.config.custom_layer_mask = None
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    model.config.custom_compensation_runtime_stats = stats
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = None
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}
                layer.self_attn.config.custom_compensation_runtime_stats = stats


def compensation_runtime_stats(model):
    stats = getattr(model.config, "custom_compensation_runtime_stats", {}) or {}
    return {
        "compensation_runtime_calls": int(stats.get("calls", 0)),
        "compensation_runtime_total_sec": float(stats.get("total_sec", 0.0)),
    }


def generate_once(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
    mask_payload,
    action_payload,
    compensation_config,
    timer: GenerationTimer,
    timer_metadata: Optional[Dict[str, object]] = None,
):
    generation_config = GenerationConfig(
        num_beams=args.top_k_items,
        length_penalty=args.length_penalty,
        num_return_sequences=args.top_k_items,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        top_k=None,
        top_p=None,
    )

    def call_generate():
        model.config.custom_compensation_runtime_stats = {"calls": 0, "total_sec": 0.0}
        if mask_payload is None:
            clear_custom_policy(model)
        else:
            set_custom_policy(
                model,
                mask_payload=mask_payload,
                action_payload=action_payload,
                compensation_config=compensation_config,
            )
        logits_processor = LogitsProcessorList(
            [
                ConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=args.top_k_items,
                    base_model=args.teacher_model,
                    eos_token_id=model.config.eos_token_id,
                )
            ]
        )
        try:
            return model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=logits_processor,
            )
        finally:
            clear_custom_policy(model)

    output = timer.measure(call_generate, samples=input_ids.size(0), metadata=timer_metadata)
    timer.add_last_metadata(compensation_runtime_stats(model))
    decoded = decode_generation(tokenizer, args.teacher_model, output.sequences, input_ids.size(1))
    return group_beam_outputs(decoded, input_ids.size(0), args.top_k_items)


def generate_once_untimed(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
):
    generation_config = GenerationConfig(
        num_beams=args.top_k_items,
        length_penalty=args.length_penalty,
        num_return_sequences=args.top_k_items,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        top_k=None,
        top_p=None,
    )
    clear_custom_policy(model)
    logits_processor = LogitsProcessorList(
        [
            ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=args.top_k_items,
                base_model=args.teacher_model,
                eos_token_id=model.config.eos_token_id,
            )
        ]
    )
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=logits_processor,
            )
    finally:
        clear_custom_policy(model)
    decoded = decode_generation(tokenizer, args.teacher_model, output.sequences, input_ids.size(1))
    return group_beam_outputs(decoded, input_ids.size(0), args.top_k_items)


def compute_quality_for_batch(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    targets: Sequence[str],
    masks: Sequence[Sequence[int]],
    actions: Sequence[Sequence[int]],
    compensation_config: Dict[str, object],
):
    scoring_ids, scoring_attention, labels = build_target_scoring_batch(
        tokenizer,
        input_ids,
        attention_mask,
        targets,
        max_len=2560,
    )
    with torch.no_grad():
        clear_custom_policy(model)
        full_logits = model(input_ids=scoring_ids, attention_mask=scoring_attention, use_cache=False).logits
        set_custom_policy(
            model,
            mask_payload=[list(map(int, mask)) for mask in masks],
            action_payload=[list(map(int, action)) for action in actions],
            compensation_config=compensation_config,
        )
        try:
            skip_logits = model(input_ids=scoring_ids, attention_mask=scoring_attention, use_cache=False).logits
        finally:
            clear_custom_policy(model)
    return quality_rows_from_logits(full_logits, skip_logits, labels)


def generate_grouped_by_mask(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
    masks: Sequence[Sequence[int]],
    actions: Sequence[Sequence[int]],
    compensation_config: Dict[str, object],
    timer: GenerationTimer,
    batch_metadata: Optional[Dict[str, object]] = None,
):
    groups = defaultdict(list)
    for idx, mask in enumerate(masks):
        action = actions[idx] if actions else [ACTION_EXECUTE if int(v) else 0 for v in mask]
        groups[(tuple(int(v) for v in mask), tuple(int(v) for v in action))].append(idx)

    grouped_outputs = [None] * len(masks)
    group_count = len(groups)
    for (mask_tuple, action_tuple), row_indices in groups.items():
        idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=input_ids.device)
        timer_metadata = dict(batch_metadata or {})
        timer_metadata.update(
            summarize_batch_masks(
                [masks[idx] for idx in row_indices],
                grouping_strategy="identical_mask_grouping",
                num_groups=group_count,
            )
        )
        sub_compensation_config = dict(compensation_config or {})
        if isinstance(sub_compensation_config.get("gates"), list):
            sub_compensation_config["gates"] = [
                sub_compensation_config["gates"][idx] for idx in row_indices
            ]
        sub_outputs = generate_once(
            model=model,
            tokenizer=tokenizer,
            args=args,
            input_ids=input_ids.index_select(0, idx_tensor),
            attention_mask=attention_mask.index_select(0, idx_tensor),
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            mask_payload=list(mask_tuple),
            action_payload=list(action_tuple),
            compensation_config=sub_compensation_config,
            timer=timer,
            timer_metadata=timer_metadata,
        )
        for local_idx, output in zip(row_indices, sub_outputs):
            grouped_outputs[local_idx] = output
    return grouped_outputs


def evaluate_oracle_batch(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
    mask_specs: Sequence[LayerMaskSpec],
    batch_indices: Sequence[int],
    ground_truths: Sequence[str],
    timer: GenerationTimer,
):
    best = {
        int(index): {
            "score": float("inf"),
            "rank": 1_000_000,
            "predictions": [],
            "mask_spec": None,
            "candidates": [],
            "quality": {},
        }
        for index in batch_indices
    }
    targets = [ground_truths[int(index)] if int(index) < len(ground_truths) else "" for index in batch_indices]
    for mask_spec in filter_masks(mask_specs, budget=args.top_k_layers):
        actions = [ACTION_EXECUTE if int(v) else 0 for v in mask_spec.mask]
        timer_metadata = summarize_batch_masks(
            [mask_spec.mask for _ in batch_indices],
            grouping_strategy="none",
            num_groups=1,
        )
        timer_metadata.update({"oracle_candidate_mask_id": mask_spec.mask_id})
        outputs = generate_once(
            model=model,
            tokenizer=tokenizer,
            args=args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            mask_payload=mask_spec.mask,
            action_payload=actions,
            compensation_config={"mode": "none", "rank": 0},
            timer=timer,
            timer_metadata=timer_metadata,
        )
        quality_rows = []
        if not args.disable_quality_metrics:
            quality_rows = compute_quality_for_batch(
                model,
                tokenizer,
                args,
                input_ids,
                attention_mask,
                targets,
                [mask_spec.mask for _ in batch_indices],
                [actions for _ in batch_indices],
                {"mode": "none", "rank": 0},
            )
        for row_pos, sample_index in enumerate(batch_indices):
            target = ground_truths[int(sample_index)] if int(sample_index) < len(ground_truths) else ""
            rank = rank_of(outputs[row_pos], target)
            candidate = {"mask_id": mask_spec.mask_id, "execution_mask": mask_spec.mask, "rank": rank}
            if quality_rows:
                candidate.update(quality_rows[row_pos])
            if args.oracle_objective == "kl" and quality_rows:
                candidate_score = float(quality_rows[row_pos]["KL_full_to_skip"])
            elif args.oracle_objective == "nll" and quality_rows:
                candidate_score = float(quality_rows[row_pos]["NLL_skip"])
            else:
                candidate_score = float(rank)
            candidate["oracle_loss"] = candidate_score
            candidate["full_loss"] = candidate.get("NLL_full")
            if args.save_oracle_candidates:
                best[int(sample_index)]["candidates"].append(candidate)
            if candidate_score < float(best[int(sample_index)]["score"]):
                best[int(sample_index)].update(
                    {
                        "score": candidate_score,
                        "rank": rank,
                        "predictions": outputs[row_pos],
                        "mask_spec": mask_spec,
                        "quality": quality_rows[row_pos] if quality_rows else {},
                    }
                )
    return best


def import_runtime_dependencies():
    global torch
    global Accelerator
    global DataLoader
    global AutoTokenizer
    global GenerationConfig
    global LogitsProcessorList
    global Qwen2ForCausalLM
    global ConstrainedLogitsProcessor
    global EvalSidDataset
    global OneLayerStudentModel
    global LayerRouter

    import torch as torch_module
    from accelerate import Accelerator as AcceleratorCls
    from torch.utils.data import DataLoader as DataLoaderCls
    from transformers import (
        AutoTokenizer as AutoTokenizerCls,
        GenerationConfig as GenerationConfigCls,
        LogitsProcessorList as LogitsProcessorListCls,
        Qwen2ForCausalLM as Qwen2ForCausalLMCls,
    )

    from LogitProcessor import ConstrainedLogitsProcessor as ConstrainedLogitsProcessorCls
    from data import EvalSidDataset as EvalSidDatasetCls
    from models.one_layer_student import OneLayerStudentModel as OneLayerStudentModelCls
    from models.router import LayerRouter as LayerRouterCls

    torch = torch_module
    Accelerator = AcceleratorCls
    DataLoader = DataLoaderCls
    AutoTokenizer = AutoTokenizerCls
    GenerationConfig = GenerationConfigCls
    LogitsProcessorList = LogitsProcessorListCls
    Qwen2ForCausalLM = Qwen2ForCausalLMCls
    ConstrainedLogitsProcessor = ConstrainedLogitsProcessorCls
    EvalSidDataset = EvalSidDatasetCls
    OneLayerStudentModel = OneLayerStudentModelCls
    LayerRouter = LayerRouterCls


def build_parser():
    parser = argparse.ArgumentParser(description="Unified OPAL-LLM evaluation for MiniOneRec/Qwen.")
    parser.add_argument(
        "--method",
        choices=["full", "static", "opal", "opal_q", "dynamic", "input_guided", "layerwise_router", "oracle"],
        required=True,
    )
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--student_ckpt", default="")
    parser.add_argument("--policy_ckpt", default="")
    parser.add_argument(
        "--allow_fallback_router",
        action="store_true",
        help="Allow OPAL to use deterministic prefix fallback scores when --policy_ckpt is absent. Debug only.",
    )
    parser.add_argument(
        "--allow_legacy_policy_ckpt",
        action="store_true",
        help="Allow OPAL to load router checkpoints without prefix-hidden metadata. Debug/ablation only.",
    )
    parser.add_argument("--mask_library", default="")
    parser.add_argument("--save_mask_library", default="")
    parser.add_argument("--mask_id", default="")
    parser.add_argument("--mask_strategies", default=",".join(DEFAULT_MASK_STRATEGIES))
    parser.add_argument("--template_library", default="", help=argparse.SUPPRESS)
    parser.add_argument("--save_template_library", default="", help=argparse.SUPPRESS)
    parser.add_argument("--template_id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--template_strategies", default="", help=argparse.SUPPRESS)
    parser.add_argument("--static_strategy", default="uniform")
    parser.add_argument("--dynamic_selection", choices=["topk_mask", "template_library"], default="topk_mask", help=argparse.SUPPRESS)
    parser.add_argument("--budget_mode", choices=["exact_topk"], default="exact_topk")
    parser.add_argument("--prefix_depth", type=int, default=4)
    parser.add_argument("--opal_stage", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--skip_rate", type=float, default=-1.0)
    parser.add_argument("--oracle_cache", default="")
    parser.add_argument("--oracle_objective", choices=["nll", "kl", "rank"], default="nll")
    parser.add_argument("--max_consecutive_skips", type=int, default=0)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--min_keep_per_stage", type=int, default=0)
    parser.add_argument("--structure_penalty", type=float, default=1.0)
    parser.add_argument("--disable_quality_metrics", action="store_true")
    parser.add_argument("--compute_full_downstream_reference", action="store_true")
    parser.add_argument("--tail_keep", type=int, default=0)
    parser.add_argument(
        "--compensation",
        choices=["none", "ungated_lowrank", "static_gate", "margin_gated"],
        default="none",
    )
    parser.add_argument("--max_compensated_skipped_layers", type=int, default=0)
    parser.add_argument("--comp_rank", type=int, default=0)
    parser.add_argument("--static_compensation_gate", type=float, default=0.5)
    parser.add_argument("--compensation_margin_delta", type=float, default=0.0)
    parser.add_argument("--compensation_margin_tau", type=float, default=1.0)
    parser.add_argument(
        "--allow_heuristic_compensation",
        action="store_true",
        help="Allow the current non-trainable ghost residual compensation implementation. Debug/ablation only.",
    )
    parser.add_argument("--layerwise_threshold", type=float, default=0.0)
    parser.add_argument("--layerwise_match_budget", action="store_true")
    parser.add_argument(
        "--input_guided_selector",
        choices=["length", "hash", "length_hash", "calibrated"],
        default="length_hash",
        help="Prompt-feature selector for the PuDDing/IG-style baseline.",
    )
    parser.add_argument(
        "--input_guided_calibration_json",
        default="",
        help="Oracle JSON with saved candidates and input_guided_features for calibrated prompt-feature selection.",
    )
    parser.add_argument(
        "--input_guided_feature_set",
        choices=["length", "hash", "length_hash"],
        default="length_hash",
        help="Prompt feature key used by the calibrated input-guided selector.",
    )
    parser.add_argument("--input_guided_num_bins", type=int, default=16)
    parser.add_argument("--budgets", default="")
    parser.add_argument("--top_k_layers", type=int, default=21)
    parser.add_argument("--top_k_items", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--timed_batches", type=int, default=0, help="0 means time every non-warmup batch")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means no evaluator-side limit")
    parser.add_argument("--random_masks_per_budget", type=int, default=0)
    parser.add_argument("--random_templates_per_budget", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--group_by_mask", action="store_true")
    parser.add_argument("--group_by_template", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save_oracle_candidates", action="store_true")
    parser.add_argument("--drop_dedup", action="store_true")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", default="./results/planrec_experiments")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.template_strategies and not args.mask_strategies:
        args.mask_strategies = args.template_strategies
    if args.random_templates_per_budget and not args.random_masks_per_budget:
        args.random_masks_per_budget = args.random_templates_per_budget
    if args.group_by_template:
        args.group_by_mask = True
    if args.comp_rank not in ALLOWED_COMPENSATION_RANKS:
        raise ValueError(f"--comp_rank must be one of {sorted(ALLOWED_COMPENSATION_RANKS)}")
    if args.max_compensated_skipped_layers not in ALLOWED_COMPENSATED_LAYERS:
        raise ValueError(
            f"--max_compensated_skipped_layers must be one of {sorted(ALLOWED_COMPENSATED_LAYERS)}"
        )
    if args.compensation != "none" and not args.allow_heuristic_compensation:
        raise ValueError(
            "Current compensation modes use a non-trainable heuristic residual, not the learnable OPAL ghost residual. "
            "Use --allow_heuristic_compensation only for ablations/debugging."
        )
    import_runtime_dependencies()

    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    dirs = result_dirs(args.output_dir)
    run_name = args.run_name or f"{args.category}_{args.method}_k{args.top_k_layers}_seed{args.seed}"

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(tokenizer, args.info_file, args.teacher_model)
    dataset_category = CATEGORY_LABELS.get(args.category, args.category)
    dataset = EvalSidDataset(
        train_file=args.test_file,
        tokenizer=tokenizer,
        max_len=2560,
        category=dataset_category,
        test=True,
        seed=args.seed,
    )
    dataloader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_left_pad(batch, tokenizer.pad_token_id),
    )
    dataloader = accelerator.prepare(dataloader)

    model = Qwen2ForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype_from_precision(args.precision),
        attn_implementation=args.attn_implementation,
    )
    model.to(device).eval()
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    num_layers = detect_num_layers(model)
    if args.method == "opal_q":
        args.top_k_layers = keep_count_from_skip_rate(num_layers, args.skip_rate, args.top_k_layers)
    effective_top_k_layers = num_layers if args.method == "full" else args.top_k_layers

    from utils_distill import get_sid_token_ids_from_info

    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    mask_specs = load_or_build_masks(args, num_layers)
    args._input_guided_calibrated_map = load_input_guided_calibration(args)
    static_mask_spec = None
    if args.method == "static":
        static_mask_spec = static_mask_from_args(args, num_layers, mask_specs)

    student = router = opal_router = None
    opal_router_metadata: Dict[str, object] = {}
    if args.method == "dynamic":
        student, router = load_student_router(args, sid_token_ids, num_layers, device)
    elif args.method == "layerwise_router" and args.student_ckpt and args.policy_ckpt:
        student, router = load_student_router(args, sid_token_ids, num_layers, device)
    oracle_cache = load_oracle_cache(args.oracle_cache) if args.oracle_cache else {}
    if args.method == "opal":
        opal_router, opal_router_metadata = load_policy_router(args, model.config.hidden_size, num_layers, device)
    elif args.method == "opal_q":
        if args.opal_stage >= 2 and not args.oracle_cache:
            raise ValueError("--oracle_cache is required for --method opal_q --opal_stage >= 2")
        if args.opal_stage >= 2 and not oracle_cache:
            raise ValueError(f"--oracle_cache has no usable oracle rows: {args.oracle_cache}")
        if args.policy_ckpt:
            opal_router, opal_router_metadata = load_policy_router(args, model.config.hidden_size, num_layers, device)
        elif not args.allow_fallback_router:
            raise ValueError("--policy_ckpt is required for --method opal_q unless --allow_fallback_router is set")
        else:
            opal_router_metadata = {"router_input_source": "deterministic_prefix_fallback"}

    timer = GenerationTimer(warmup_batches=args.warmup_batches, timed_batches=args.timed_batches)
    component_timer = ComponentTimer()
    ground_truths = load_ground_truths(args.test_file, drop_dedup=args.drop_dedup)
    valid_sids = load_valid_sids(args.info_file)
    predictions = []

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "run_name": run_name,
                    "method": args.method,
                    "num_layers": num_layers,
                    "top_k_layers": effective_top_k_layers,
                    "num_masks": len(mask_specs),
                    "device": str(device),
                },
                indent=2,
            )
        )

    for batch_id, batch in enumerate(dataloader):
        if args.max_batches and batch_id >= args.max_batches:
            break
        if timer.should_stop():
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_indices = [int(x) for x in batch["index"].detach().cpu().tolist()]
        batch_size = input_ids.size(0)
        batch_is_warmup = timer.next_is_warmup()
        batch_targets = [
            ground_truths[int(index)] if int(index) < len(ground_truths) else ""
            for index in batch_indices
        ]
        quality_rows = None
        full_reference_outputs = None

        if args.method == "full":
            masks = [[1] * num_layers for _ in range(batch_size)]
            mask_ids = ["full_teacher"] * batch_size
            action_plans = build_actions_for_batch(args, masks, [[1.0] * num_layers for _ in masks])
            actions = [plan["action_mask"] for plan in action_plans]
            batch_metadata = summarize_batch_masks(masks, grouping_strategy="none", num_groups=1)
            outputs = generate_once(
                model,
                tokenizer,
                args,
                input_ids,
                attention_mask,
                prefix_allowed_tokens_fn,
                None,
                None,
                {"mode": "none", "rank": 0},
                timer,
                timer_metadata=batch_metadata,
            )
        elif args.method == "static":
            masks = [static_mask_spec.mask for _ in range(batch_size)]
            mask_ids = [static_mask_spec.mask_id for _ in range(batch_size)]
            action_plans = timed_action_plans(args, masks, None, component_timer, metadata={"warmup": batch_is_warmup})
            actions = [plan["action_mask"] for plan in action_plans]
            comp_config = compensation_config_from_args(args, action_plans)
            batch_metadata = summarize_batch_masks(masks, grouping_strategy="none", num_groups=1)
            outputs = generate_once(
                model,
                tokenizer,
                args,
                input_ids,
                attention_mask,
                prefix_allowed_tokens_fn,
                static_mask_spec.mask,
                actions[0],
                comp_config,
                timer,
                timer_metadata=batch_metadata,
            )
        elif args.method == "dynamic":
            (masks, mask_ids, router_scores), _ = component_timer.measure(
                "router",
                lambda: dynamic_masks(args, student, router, input_ids, attention_mask, num_layers),
                samples=batch_size,
                metadata={"router_source": "one_layer_student_policy", "warmup": batch_is_warmup},
            )
            action_plans = timed_action_plans(args, masks, router_scores, component_timer, metadata={"warmup": batch_is_warmup})
            actions = [plan["action_mask"] for plan in action_plans]
            comp_config = compensation_config_from_args(args, action_plans)
            if args.group_by_mask:
                outputs = generate_grouped_by_mask(
                    model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, actions, comp_config, timer
                )
            else:
                batch_metadata = summarize_batch_masks(masks, grouping_strategy="layerwise_subset", num_groups=1)
                outputs = generate_once(
                    model,
                    tokenizer,
                    args,
                    input_ids,
                    attention_mask,
                    prefix_allowed_tokens_fn,
                    masks,
                    actions,
                    comp_config,
                    timer,
                    timer_metadata=batch_metadata,
                )
        else:
            if args.method == "opal":
                masks, mask_ids, router_scores, _, _ = opal_prefix_masks(
                    args, model, opal_router, input_ids, attention_mask, num_layers, component_timer, is_warmup=batch_is_warmup
                )
                action_plans = timed_action_plans(args, masks, router_scores, component_timer, metadata={"warmup": batch_is_warmup})
                actions = [plan["action_mask"] for plan in action_plans]
                comp_config = compensation_config_from_args(args, action_plans)
                if args.group_by_mask:
                    outputs = generate_grouped_by_mask(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                    )
                else:
                    batch_metadata = summarize_batch_masks(masks, grouping_strategy="layerwise_subset", num_groups=1)
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                        timer_metadata=batch_metadata,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    predictions.append(
                        {
                            "index": int(sample_index),
                            "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                            "sample_predictions": outputs[local_pos],
                            "layer_mask": masks[local_pos],
                            "execution_mask": masks[local_pos],
                            "action_mask": actions[local_pos],
                            "mask_id": mask_ids[local_pos],
                            "action_id": action_plans[local_pos]["action_id"],
                            "kept_layer_count": sum(masks[local_pos]),
                            "prefix_depth": args.prefix_depth,
                            "layer_scores": router_scores[local_pos],
                            "compensation_mask": action_plans[local_pos]["compensation_mask"],
                            "compensation_gates": action_plans[local_pos]["compensation_gates"],
                            "compensated_layer_count": action_plans[local_pos]["compensated_layer_count"],
                        }
                )
                continue

            if args.method == "opal_q":
                masks, mask_ids, keep_scores, skip_risks, opal_q_rows = opal_q_masks(
                    args,
                    model,
                    opal_router,
                    oracle_cache,
                    input_ids,
                    attention_mask,
                    batch_indices,
                    num_layers,
                    component_timer,
                    is_warmup=batch_is_warmup,
                )
                action_plans = timed_action_plans(args, masks, skip_risks, component_timer, metadata={"warmup": batch_is_warmup})
                actions = [plan["action_mask"] for plan in action_plans]
                comp_config = compensation_config_from_args(args, action_plans)
                if args.group_by_mask:
                    outputs = generate_grouped_by_mask(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                    )
                else:
                    batch_metadata = summarize_batch_masks(masks, grouping_strategy="layerwise_subset", num_groups=1)
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                        timer_metadata=batch_metadata,
                    )
                if not args.disable_quality_metrics:
                    quality_rows = compute_quality_for_batch(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        batch_targets,
                        masks,
                        actions,
                        comp_config,
                    )
                if args.compute_full_downstream_reference or args.method == "opal_q":
                    full_reference_outputs = generate_once_untimed(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    record = {
                        "index": int(sample_index),
                        "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                        "sample_predictions": outputs[local_pos],
                        "full_sample_predictions": full_reference_outputs[local_pos] if full_reference_outputs else [],
                        "layer_mask": masks[local_pos],
                        "execution_mask": masks[local_pos],
                        "action_mask": actions[local_pos],
                        "mask_id": mask_ids[local_pos],
                        "action_id": action_plans[local_pos]["action_id"],
                        "compensation_mask": action_plans[local_pos]["compensation_mask"],
                        "compensation_gates": action_plans[local_pos]["compensation_gates"],
                        "compensated_layer_count": action_plans[local_pos]["compensated_layer_count"],
                    }
                    record.update(opal_q_rows[local_pos])
                    if quality_rows:
                        record.update(quality_rows[local_pos])
                        if record.get("mask_regret") is None and record.get("oracle_loss") is not None:
                            record["mask_regret"] = record["NLL_skip"] - float(record["oracle_loss"])
                    if record.get("mask_regret") is not None:
                        record["oracle_regret"] = record["mask_regret"]
                    predictions.append(record)
                continue

            if args.method == "input_guided":
                (mask_result, _router_time) = component_timer.measure(
                    "router",
                    lambda: input_guided_masks(args, input_ids, attention_mask, num_layers),
                    samples=batch_size,
                    metadata={"router_source": "prompt_features", "warmup": batch_is_warmup},
                )
                masks, mask_ids, input_guided_features, router_scores = mask_result
                action_plans = timed_action_plans(args, masks, router_scores, component_timer, metadata={"warmup": batch_is_warmup})
                actions = [plan["action_mask"] for plan in action_plans]
                comp_config = compensation_config_from_args(args, action_plans)
                if args.group_by_mask:
                    outputs = generate_grouped_by_mask(
                        model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, actions, comp_config, timer
                    )
                else:
                    batch_metadata = summarize_batch_masks(masks, grouping_strategy="layerwise_subset", num_groups=1)
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                        timer_metadata=batch_metadata,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    predictions.append(
                        {
                            "index": int(sample_index),
                            "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                            "sample_predictions": outputs[local_pos],
                            "layer_mask": masks[local_pos],
                            "execution_mask": masks[local_pos],
                            "action_mask": actions[local_pos],
                            "mask_id": mask_ids[local_pos],
                            "action_id": action_plans[local_pos]["action_id"],
                            "layer_scores": router_scores[local_pos],
                            "kept_layer_count": sum(masks[local_pos]),
                            "input_guided_features": input_guided_features[local_pos],
                            "compensation_mask": action_plans[local_pos]["compensation_mask"],
                            "compensation_gates": action_plans[local_pos]["compensation_gates"],
                            "compensated_layer_count": action_plans[local_pos]["compensated_layer_count"],
                        }
                    )
                continue

            if args.method == "layerwise_router":
                (masks, mask_ids, router_scores), _ = component_timer.measure(
                    "router",
                    lambda: layerwise_router_masks(args, student, router, input_ids, attention_mask, num_layers),
                    samples=batch_size,
                    metadata={
                        "router_source": "one_layer_student_local_threshold"
                        if student is not None and router is not None
                        else "prompt_feature_local_threshold_fallback",
                        "warmup": batch_is_warmup,
                    },
                )
                action_plans = timed_action_plans(args, masks, router_scores, component_timer, metadata={"warmup": batch_is_warmup})
                actions = [plan["action_mask"] for plan in action_plans]
                comp_config = compensation_config_from_args(args, action_plans)
                if args.group_by_mask:
                    outputs = generate_grouped_by_mask(
                        model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, actions, comp_config, timer
                    )
                else:
                    batch_metadata = summarize_batch_masks(masks, grouping_strategy="layerwise_subset", num_groups=1)
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        actions,
                        comp_config,
                        timer,
                        timer_metadata=batch_metadata,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    predictions.append(
                        {
                            "index": int(sample_index),
                            "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                            "sample_predictions": outputs[local_pos],
                            "layer_mask": masks[local_pos],
                            "execution_mask": masks[local_pos],
                            "action_mask": actions[local_pos],
                            "mask_id": mask_ids[local_pos],
                            "action_id": action_plans[local_pos]["action_id"],
                            "router_scores": router_scores[local_pos],
                            "layer_scores": router_scores[local_pos],
                            "kept_layer_count": sum(masks[local_pos]),
                            "compensation_mask": action_plans[local_pos]["compensation_mask"],
                            "compensation_gates": action_plans[local_pos]["compensation_gates"],
                            "compensated_layer_count": action_plans[local_pos]["compensated_layer_count"],
                        }
                    )
                continue

            oracle_rows = evaluate_oracle_batch(
                model,
                tokenizer,
                args,
                input_ids,
                attention_mask,
                prefix_allowed_tokens_fn,
                mask_specs,
                batch_indices,
                ground_truths,
                timer,
            )
            for local_pos, sample_index in enumerate(batch_indices):
                feature = input_guided_features(
                    input_ids[local_pos : local_pos + 1],
                    attention_mask[local_pos : local_pos + 1],
                    args.input_guided_num_bins,
                )[0]
                row = oracle_rows[int(sample_index)]
                mask_spec = row["mask_spec"]
                selected_mask = mask_spec.mask if mask_spec else []
                quality = dict(row.get("quality") or {})
                record = {
                    "index": int(sample_index),
                    "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                    "sample_predictions": row["predictions"],
                    "layer_mask": selected_mask,
                    "execution_mask": selected_mask,
                    "oracle_mask": selected_mask,
                    "action_mask": [ACTION_EXECUTE if int(v) else 0 for v in selected_mask],
                    "mask_id": mask_spec.mask_id if mask_spec else "none",
                    "oracle_rank": row["rank"],
                    "oracle_loss": row["score"],
                    "full_loss": quality.get("NLL_full"),
                    "mask_regret": 0.0,
                    "oracle_regret": 0.0,
                    "oracle_candidates": row["candidates"] if args.save_oracle_candidates else [],
                    "input_guided_features": summarize_input_guided_feature(
                        feature, args.input_guided_feature_set
                    ),
                }
                record.update(quality)
                predictions.append(record)
            continue

        for local_pos, sample_index in enumerate(batch_indices):
            record = {
                "index": int(sample_index),
                "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                "sample_predictions": outputs[local_pos],
                "layer_mask": masks[local_pos],
                "execution_mask": masks[local_pos],
                "action_mask": actions[local_pos],
                "mask_id": mask_ids[local_pos],
                "action_id": action_plans[local_pos]["action_id"],
                "kept_layer_count": sum(masks[local_pos]),
                "compensation_mask": action_plans[local_pos]["compensation_mask"],
                "compensation_gates": action_plans[local_pos]["compensation_gates"],
                "compensated_layer_count": action_plans[local_pos]["compensated_layer_count"],
            }
            if args.method == "dynamic":
                record["router_scores"] = router_scores[local_pos]
                record["layer_scores"] = router_scores[local_pos]
            predictions.append(record)

    rank_path = dirs["raw_json"] / f"{run_name}_rank{accelerator.process_index}.json"
    write_json(
        rank_path,
        {
            "predictions": predictions,
            "timing_records": timer.raw_records(),
            "component_timing_records": component_timer.raw_records(),
        },
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_predictions = []
        timing_records = []
        component_timing_records = []
        for path in sorted(dirs["raw_json"].glob(f"{run_name}_rank*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_predictions.extend(payload.get("predictions", []))
            timing_records.extend(payload.get("timing_records", []))
            component_timing_records.extend(payload.get("component_timing_records", []))
        all_predictions = sorted(all_predictions, key=lambda row: int(row.get("index", 0)))

        metadata = {
            "dataset": args.category,
            "method": args.method,
            "top_k_layers": effective_top_k_layers,
            "checkpoint": args.teacher_model,
            "student_ckpt": args.student_ckpt,
            "policy_ckpt": args.policy_ckpt,
            "policy_ckpt_metadata": opal_router_metadata if args.method in {"opal", "opal_q"} else {},
            "allow_fallback_router": args.allow_fallback_router,
            "allow_legacy_policy_ckpt": args.allow_legacy_policy_ckpt,
            "mask_library": args.mask_library or args.save_mask_library or args.template_library or args.save_template_library,
            "mask_id": args.mask_id or args.template_id,
            "static_strategy": args.static_strategy,
            "budget_mode": args.budget_mode,
            "skip_rate": actual_skip_rate([1] * effective_top_k_layers + [0] * max(0, num_layers - effective_top_k_layers))
            if args.method != "full"
            else 0.0,
            "requested_skip_rate": args.skip_rate if args.skip_rate >= 0 else None,
            "opal_stage": args.opal_stage if args.method == "opal_q" else None,
            "prefix_depth": args.prefix_depth,
            "oracle_cache": args.oracle_cache,
            "oracle_objective": args.oracle_objective,
            "oracle_cache_size": len(oracle_cache),
            "max_consecutive_skips": args.max_consecutive_skips,
            "num_stages": args.num_stages,
            "min_keep_per_stage": args.min_keep_per_stage,
            "structure_penalty": args.structure_penalty,
            "quality_metrics_enabled": not args.disable_quality_metrics,
            "full_downstream_reference": args.compute_full_downstream_reference or args.method == "opal_q",
            "tail_keep": args.tail_keep,
            "compensation": args.compensation,
            "compensation_impl": "heuristic_non_trainable" if args.compensation != "none" else "none",
            "allow_heuristic_compensation": args.allow_heuristic_compensation,
            "max_compensated_skipped_layers": args.max_compensated_skipped_layers,
            "comp_rank": args.comp_rank,
            "layerwise_threshold": args.layerwise_threshold,
            "layerwise_match_budget": args.layerwise_match_budget,
            "dynamic_selection": args.dynamic_selection,
            "input_guided_selector": args.input_guided_selector,
            "input_guided_calibration_json": args.input_guided_calibration_json,
            "input_guided_feature_set": args.input_guided_feature_set,
            "input_guided_num_bins": args.input_guided_num_bins,
            "input_guided_calibrated_keys": len(args._input_guided_calibrated_map),
            "group_by_mask": args.group_by_mask,
            "seed": args.seed,
            "command": command_line(),
            "git_commit": git_commit(current_dir),
            "num_layers": num_layers,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "attention_backend": args.attn_implementation,
            "beam_size": args.top_k_items,
            "max_new_tokens": args.max_new_tokens,
            "length_penalty": args.length_penalty,
            "drop_dedup": args.drop_dedup,
            "oracle_selection_cost_included": args.method == "oracle",
        }
        # Recompute latency summary from raw rank files across all accelerator processes.
        timed = [row for row in timing_records if not row.get("warmup")]
        if timed:
            durations = sorted(float(row["duration_sec"]) for row in timed)
            total_time = sum(durations)
            total_samples = sum(int(row["samples"]) for row in timed)
            mean = total_time / len(durations)
            var = sum((value - mean) ** 2 for value in durations) / len(durations)

            def percentile(p):
                if len(durations) == 1:
                    return durations[0]
                pos = (len(durations) - 1) * p
                lower = int(pos)
                upper = min(len(durations) - 1, lower + 1)
                weight = pos - lower
                return durations[lower] * (1 - weight) + durations[upper] * weight

            latency_summary = {
                "warmup_batches": args.warmup_batches,
                "timed_batches": len(timed),
                "latency_mean_sec": mean,
                "latency_std_sec": var ** 0.5,
                "latency_p50_sec": percentile(0.50),
                "latency_p95_sec": percentile(0.95),
                "throughput_samples_per_sec": total_samples / total_time if total_time > 0 else 0.0,
                "physical_generate_latency_sec": mean,
            }
        else:
            latency_summary = timer.summary()
            latency_summary["physical_generate_latency_sec"] = latency_summary.get("latency_mean_sec", 0.0)

        component_groups: Dict[str, List[float]] = defaultdict(list)
        for record in component_timing_records:
            if bool((record.get("metadata") or {}).get("warmup", False)):
                continue
            component_groups[str(record.get("name", ""))].append(float(record.get("duration_sec", 0.0)))
        for component_name, values in component_groups.items():
            if not component_name:
                continue
            latency_summary[f"{component_name}_latency_sec"] = sum(values) / len(values) if values else 0.0
            latency_summary[f"{component_name}_latency_total_sec"] = sum(values)
        latency_summary.setdefault("router_latency_sec", 0.0)
        latency_summary.setdefault("prefix_latency_sec", 0.0)
        latency_summary.setdefault("compensation_latency_sec", 0.0)
        latency_summary.setdefault("grouping_overhead_sec", 0.0)
        timed_component_records = [
            record
            for record in component_timing_records
            if not bool((record.get("metadata") or {}).get("warmup", False))
        ]
        component_overhead_total = sum(float(record.get("duration_sec", 0.0)) for record in timed_component_records)
        timed_generate_records = [row for row in timing_records if not row.get("warmup")]
        latency_summary["component_overhead_total_sec"] = component_overhead_total
        latency_summary["component_overhead_records"] = len(timed_component_records)
        if timed_generate_records:
            latency_summary["physical_generate_latency_mean_sec"] = latency_summary.get("physical_generate_latency_sec", 0.0)
            latency_summary["component_overhead_mean_sec"] = component_overhead_total / max(1, len(timed_generate_records))
        runtime_comp = [
            float((row.get("metadata") or {}).get("compensation_runtime_total_sec", 0.0))
            for row in timed_generate_records
        ]
        latency_summary["compensation_runtime_total_sec"] = sum(runtime_comp)
        if latency_summary.get("latency_mean_sec", 0.0) > 0:
            latency_summary["compensation_overhead_ratio"] = latency_summary["compensation_runtime_total_sec"] / max(
                1e-12, sum(float(row.get("duration_sec", 0.0)) for row in timing_records)
            )
        else:
            latency_summary["compensation_overhead_ratio"] = 0.0

        final_payload = build_payload(
            metadata=metadata,
            predictions=all_predictions,
            ground_truths=ground_truths,
            valid_sids=valid_sids,
            latency=latency_summary,
            timing_records=timing_records,
        )
        final_payload["component_timing_records"] = component_timing_records
        final_path = dirs["raw_json"] / f"{run_name}.json"
        write_json(final_path, final_payload)
        append_summary_csv(dirs["tables"] / "summary.csv", summary_row(final_payload, str(final_path)))
        write_json(dirs["logs"] / f"{run_name}_metadata.json", metadata)
        print(f"Saved final OPAL result JSON to {final_path}")
        print(f"Updated summary table at {dirs['tables'] / 'summary.csv'}")


if __name__ == "__main__":
    main()
