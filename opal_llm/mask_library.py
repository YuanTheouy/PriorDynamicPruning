import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ACTION_SKIP = 0
ACTION_EXECUTE = 1
ACTION_COMPENSATE = 2

ALLOWED_COMPENSATION_RANKS = {0, 1, 2, 4}
ALLOWED_COMPENSATED_LAYERS = {0, 1, 2, 4}

DEFAULT_MASK_STRATEGIES = (
    "uniform",
    "first_k",
    "last_k",
    "ends_heavy",
    "middle_heavy",
    "shortgpt",
)


@dataclass(frozen=True)
class LayerMaskSpec:
    mask_id: str
    strategy: str
    budget: int
    num_layers: int
    mask: List[int]
    metadata: Dict[str, object]


def _clamp_budget(num_layers: int, budget: int) -> int:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return max(0, min(int(budget), int(num_layers)))


def _fill_unique(indices: Iterable[int], num_layers: int, budget: int) -> List[int]:
    selected = []
    seen = set()
    for idx in indices:
        idx = max(0, min(num_layers - 1, int(idx)))
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) == budget:
            return selected

    for idx in range(num_layers):
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) == budget:
            break
    return selected


def mask_key(mask: Sequence[int]) -> str:
    return "".join(str(int(v)) for v in mask)


def mask_id_from_mask(mask: Sequence[int], prefix: str = "mask") -> str:
    digest = hashlib.sha1(mask_key(mask).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_k{sum(int(v) for v in mask)}_{digest}"


def action_id_from_actions(actions: Sequence[int]) -> str:
    digest = hashlib.sha1("".join(str(int(v)) for v in actions).encode("utf-8")).hexdigest()[:10]
    execute_count = sum(1 for v in actions if int(v) == ACTION_EXECUTE)
    compensate_count = sum(1 for v in actions if int(v) == ACTION_COMPENSATE)
    return f"action_e{execute_count}_c{compensate_count}_{digest}"


def generate_mask(strategy: str, num_layers: int, budget: int, seed: int = 0, variant: int = 0) -> List[int]:
    """Return a binary decoder-layer execution mask."""
    budget = _clamp_budget(num_layers, budget)
    mask = [0] * num_layers
    if budget == 0:
        return mask
    if budget == num_layers:
        return [1] * num_layers

    if strategy == "uniform":
        if budget == 1:
            indices = [num_layers // 2]
        else:
            indices = [round(i * (num_layers - 1) / (budget - 1)) for i in range(budget)]
        indices = _fill_unique(indices, num_layers, budget)
    elif strategy == "first_k":
        indices = list(range(budget))
    elif strategy == "last_k":
        indices = list(range(num_layers - budget, num_layers))
    elif strategy == "ends_heavy":
        left = (budget + 1) // 2
        right = budget - left
        indices = list(range(left)) + list(range(num_layers - right, num_layers))
    elif strategy == "middle_heavy":
        start = (num_layers - budget) // 2
        indices = list(range(start, start + budget))
    elif strategy == "shortgpt":
        # Position-only ShortGPT-style proxy: preserve boundary blocks and prune a
        # contiguous middle band when no calibrated layer influence file is given.
        left = max(1, budget // 3)
        right = budget - left
        indices = list(range(left)) + list(range(num_layers - right, num_layers))
    elif strategy == "random_diverse":
        rng = random.Random(seed + variant * 1009 + budget * 9173 + num_layers)
        indices = sorted(rng.sample(range(num_layers), budget))
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")

    for idx in indices:
        mask[idx] = 1
    return mask


def _strategy_mask_id(strategy: str, budget: int, variant: int, mask: Sequence[int]) -> str:
    if strategy == "random_diverse":
        return f"{strategy}_k{budget}_v{variant}"
    if strategy in DEFAULT_MASK_STRATEGIES:
        return f"{strategy}_k{budget}"
    return mask_id_from_mask(mask, prefix=strategy)


def generate_mask_library(
    num_layers: int,
    budgets: Sequence[int],
    strategies: Sequence[str] = DEFAULT_MASK_STRATEGIES,
    random_masks_per_budget: int = 0,
    seed: int = 0,
) -> List[LayerMaskSpec]:
    specs: List[LayerMaskSpec] = []
    seen_masks = set()
    expanded: List[Tuple[str, int]] = [(strategy, 0) for strategy in strategies]
    expanded.extend(("random_diverse", i) for i in range(random_masks_per_budget))

    for budget in budgets:
        for strategy, variant in expanded:
            mask = generate_mask(strategy, num_layers, budget, seed=seed, variant=variant)
            key = tuple(mask)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            clamped_budget = _clamp_budget(num_layers, budget)
            specs.append(
                LayerMaskSpec(
                    mask_id=_strategy_mask_id(strategy, clamped_budget, variant, mask),
                    strategy=strategy,
                    budget=sum(mask),
                    num_layers=num_layers,
                    mask=mask,
                    metadata={"seed": seed, "variant": variant},
                )
            )
    return specs


def save_mask_library(path: str, masks: Sequence[LayerMaskSpec]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "schema": "opal_layer_masks",
        "masks": [asdict(mask) for mask in masks],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mask_library(path: str) -> List[LayerMaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("masks", payload.get("templates", payload if isinstance(payload, list) else []))
    masks = []
    for row in rows:
        mask = [int(v) for v in row["mask"]]
        masks.append(
            LayerMaskSpec(
                mask_id=str(row.get("mask_id") or row.get("template_id") or mask_id_from_mask(mask)),
                strategy=str(row.get("strategy", "unknown")),
                budget=int(row.get("budget", sum(mask))),
                num_layers=int(row.get("num_layers", len(mask))),
                mask=mask,
                metadata=dict(row.get("metadata", {})),
            )
        )
    return masks


def filter_masks(
    masks: Sequence[LayerMaskSpec],
    budget: Optional[int] = None,
    mask_id: Optional[str] = None,
) -> List[LayerMaskSpec]:
    selected = list(masks)
    if budget is not None:
        selected = [mask for mask in selected if mask.budget == int(budget)]
    if mask_id:
        selected = [mask for mask in selected if mask.mask_id == mask_id]
    if not selected:
        detail = f"budget={budget}, mask_id={mask_id}"
        raise ValueError(f"No masks matched {detail}")
    return selected


def assignment_entropy(ids: Sequence[str]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    total = float(sum(counts.values()))
    entropy = -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())
    return max(0.0, entropy)


def mask_entropy(masks: Sequence[Sequence[int]]) -> float:
    return assignment_entropy([mask_key(mask) for mask in masks])


def summarize_masks(masks: Sequence[Sequence[int]], mask_ids: Optional[Sequence[str]] = None) -> Dict[str, object]:
    if not masks:
        return {
            "average_kept_layers": 0.0,
            "unique_masks": 0,
            "mask_entropy": 0.0,
            "mask_usage": {},
            "mask_id_entropy": 0.0,
        }

    kept = [sum(int(v) for v in mask) for mask in masks]
    keys = [mask_key(mask) for mask in masks]
    usage = dict(Counter(mask_ids or keys))
    num_layers = len(masks[0]) if masks and masks[0] else 0
    layer_usage = []
    if num_layers:
        for idx in range(num_layers):
            layer_usage.append(sum(int(mask[idx]) for mask in masks) / float(len(masks)))
    return {
        "average_kept_layers": sum(kept) / len(kept),
        "unique_masks": len(set(keys)),
        "mask_entropy": mask_entropy(masks),
        "mask_usage": usage,
        "mask_id_entropy": assignment_entropy(list(mask_ids or keys)),
        "layer_usage": layer_usage,
    }


def summarize_batch_masks(
    masks: Sequence[Sequence[int]],
    grouping_strategy: str,
    num_groups: Optional[int] = None,
) -> Dict[str, object]:
    keys = [mask_key(mask) for mask in masks]
    unique = len(set(keys))
    return {
        "batch_size": len(masks),
        "unique_masks_in_batch": unique,
        "mask_entropy_in_batch": assignment_entropy(keys),
        "grouping_strategy": grouping_strategy,
        "num_groups": int(num_groups if num_groups is not None else max(1, unique)),
    }


def exact_topk_mask_from_scores(
    scores,
    top_k: int,
    prefix_depth: int = 0,
    tail_keep: int = 0,
):
    import torch

    score_tensor = scores.detach().float()
    if score_tensor.dim() == 1:
        score_tensor = score_tensor.unsqueeze(0)
    batch_size, num_layers = score_tensor.shape
    top_k = _clamp_budget(num_layers, top_k)
    forced = set(range(min(prefix_depth, num_layers)))
    if tail_keep > 0:
        forced.update(range(max(0, num_layers - int(tail_keep)), num_layers))
    if len(forced) > top_k:
        raise ValueError(
            f"Forced prefix/tail layers ({len(forced)}) exceed exact top-k budget ({top_k}). "
            "Lower --prefix_depth/--tail_keep or increase --top_k_layers."
        )

    masks = torch.zeros_like(score_tensor)
    if forced:
        forced_idx = torch.tensor(sorted(forced), device=score_tensor.device, dtype=torch.long)
        masks[:, forced_idx] = 1.0

    remaining_budget = max(0, top_k - len(forced))
    if remaining_budget > 0:
        candidate_scores = score_tensor.clone()
        if forced:
            candidate_scores[:, forced_idx] = -float("inf")
        _, top_indices = torch.topk(candidate_scores, remaining_budget, dim=-1)
        masks.scatter_(1, top_indices, 1.0)
    return masks


def local_threshold_mask_from_scores(
    scores,
    threshold: float,
    top_k: int,
    match_budget: bool,
):
    import torch

    score_tensor = scores.detach().float()
    if score_tensor.dim() == 1:
        score_tensor = score_tensor.unsqueeze(0)
    if match_budget:
        return exact_topk_mask_from_scores(score_tensor, top_k=top_k)
    mask = (score_tensor >= float(threshold)).float()
    empty_rows = torch.where(mask.sum(dim=1) == 0)[0]
    if empty_rows.numel() > 0:
        best = torch.argmax(score_tensor[empty_rows], dim=1, keepdim=True)
        mask[empty_rows] = 0.0
        mask[empty_rows].scatter_(1, best, 1.0)
    return mask


def tensor_masks_to_lists(mask_tensor) -> List[List[int]]:
    return [[int(round(float(v))) for v in row] for row in mask_tensor.detach().float().cpu().tolist()]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def build_action_plan(
    execution_mask: Sequence[int],
    scores: Sequence[float],
    compensation: str,
    max_compensated_skipped_layers: int,
    margin_delta: float,
    margin_tau: float,
    static_gate: float,
) -> Dict[str, object]:
    actions = [ACTION_EXECUTE if int(v) else ACTION_SKIP for v in execution_mask]
    compensation_mask = [0] * len(execution_mask)
    gates = [0.0] * len(execution_mask)
    if compensation == "none" or int(max_compensated_skipped_layers) <= 0:
        return {
            "action_mask": actions,
            "action_id": action_id_from_actions(actions),
            "compensation_mask": compensation_mask,
            "compensation_gates": gates,
            "compensated_layer_count": 0,
        }

    executed_scores = [float(score) for keep, score in zip(execution_mask, scores) if int(keep)]
    threshold = min(executed_scores) if executed_scores else max(float(score) for score in scores)
    skipped = [
        (idx, float(score), abs(float(score) - threshold))
        for idx, (keep, score) in enumerate(zip(execution_mask, scores))
        if not int(keep)
    ]
    skipped = sorted(skipped, key=lambda row: (-row[1], row[2], row[0]))
    selected = skipped[: int(max_compensated_skipped_layers)]

    for idx, score, _ in selected:
        actions[idx] = ACTION_COMPENSATE
        compensation_mask[idx] = 1
        if compensation == "ungated_lowrank":
            gate = 1.0
        elif compensation == "static_gate":
            gate = float(static_gate)
        elif compensation == "margin_gated":
            tau = max(float(margin_tau), 1e-6)
            gate = _sigmoid((float(score) - float(threshold) + float(margin_delta)) / tau)
        else:
            raise ValueError(f"Unknown compensation mode: {compensation}")
        gates[idx] = float(max(0.0, min(1.0, gate)))

    return {
        "action_mask": actions,
        "action_id": action_id_from_actions(actions),
        "compensation_mask": compensation_mask,
        "compensation_gates": gates,
        "compensated_layer_count": sum(compensation_mask),
    }


def prompt_feature_scores(features: Sequence[Dict[str, int]], num_layers: int, seed: int, feature_set: str):
    import torch

    rows = []
    for feature in features:
        length_term = int(feature.get("length_bin", 0)) + 1
        hash_term = int(feature.get("token_hash_bin", 0)) + int(seed) + 1
        last_term = int(feature.get("last_token_bin", 0)) + 1
        scores = []
        for layer in range(num_layers):
            pos = (layer + 1) / max(1, num_layers)
            score = 0.10 * pos
            if feature_set in {"length", "length_hash"}:
                score += math.cos((layer + 1) * length_term * 0.37)
            if feature_set in {"hash", "length_hash"}:
                score += math.sin((layer + 3) * hash_term * 0.013)
                score += 0.25 * math.cos((layer + 5) * last_term * 0.19)
            scores.append(score)
        rows.append(scores)
    return torch.tensor(rows, dtype=torch.float32)
