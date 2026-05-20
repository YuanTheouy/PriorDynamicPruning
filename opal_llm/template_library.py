import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TEMPLATE_STRATEGIES = (
    "uniform",
    "first_k",
    "last_k",
    "ends_heavy",
    "middle_heavy",
)


@dataclass(frozen=True)
class LayerTemplate:
    template_id: str
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


def generate_mask(strategy: str, num_layers: int, budget: int, seed: int = 0, variant: int = 0) -> List[int]:
    """Return a binary layer-retention mask for a finite OPAL template."""
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
    elif strategy == "random_diverse":
        rng = random.Random(seed + variant * 1009 + budget * 9173 + num_layers)
        indices = sorted(rng.sample(range(num_layers), budget))
    else:
        raise ValueError(f"Unknown template strategy: {strategy}")

    for idx in indices:
        mask[idx] = 1
    return mask


def _template_id(strategy: str, budget: int, variant: int) -> str:
    if strategy == "random_diverse":
        return f"{strategy}_k{budget}_v{variant}"
    return f"{strategy}_k{budget}"


def generate_template_library(
    num_layers: int,
    budgets: Sequence[int],
    strategies: Sequence[str] = DEFAULT_TEMPLATE_STRATEGIES,
    random_templates_per_budget: int = 0,
    seed: int = 0,
) -> List[LayerTemplate]:
    templates: List[LayerTemplate] = []
    seen_masks = set()

    expanded: List[Tuple[str, int]] = [(strategy, 0) for strategy in strategies]
    expanded.extend(("random_diverse", i) for i in range(random_templates_per_budget))

    for budget in budgets:
        for strategy, variant in expanded:
            mask = generate_mask(strategy, num_layers, budget, seed=seed, variant=variant)
            key = tuple(mask)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            templates.append(
                LayerTemplate(
                    template_id=_template_id(strategy, _clamp_budget(num_layers, budget), variant),
                    strategy=strategy,
                    budget=sum(mask),
                    num_layers=num_layers,
                    mask=mask,
                    metadata={"seed": seed, "variant": variant},
                )
            )
    return templates


def save_template_library(path: str, templates: Sequence[LayerTemplate]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "templates": [asdict(template) for template in templates],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_template_library(path: str) -> List[LayerTemplate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("templates", payload if isinstance(payload, list) else [])
    templates = []
    for row in rows:
        templates.append(
            LayerTemplate(
                template_id=str(row["template_id"]),
                strategy=str(row.get("strategy", "unknown")),
                budget=int(row.get("budget", sum(row["mask"]))),
                num_layers=int(row.get("num_layers", len(row["mask"]))),
                mask=[int(v) for v in row["mask"]],
                metadata=dict(row.get("metadata", {})),
            )
        )
    return templates


def filter_templates(
    templates: Sequence[LayerTemplate],
    budget: Optional[int] = None,
    template_id: Optional[str] = None,
) -> List[LayerTemplate]:
    selected = list(templates)
    if budget is not None:
        selected = [template for template in selected if template.budget == int(budget)]
    if template_id:
        selected = [template for template in selected if template.template_id == template_id]
    if not selected:
        detail = f"budget={budget}, template_id={template_id}"
        raise ValueError(f"No templates matched {detail}")
    return selected


def exact_template_lookup(templates: Sequence[LayerTemplate]) -> Dict[Tuple[int, ...], LayerTemplate]:
    return {tuple(template.mask): template for template in templates}


def assignment_entropy(template_ids: Sequence[str]) -> float:
    if not template_ids:
        return 0.0
    counts = Counter(template_ids)
    total = float(sum(counts.values()))
    entropy = -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())
    return max(0.0, entropy)


def mask_entropy(masks: Sequence[Sequence[int]]) -> float:
    keys = ["".join(str(int(v)) for v in mask) for mask in masks]
    return assignment_entropy(keys)


def summarize_masks(masks: Sequence[Sequence[int]], template_ids: Optional[Sequence[str]] = None) -> Dict[str, object]:
    if not masks:
        return {
            "average_kept_layers": 0.0,
            "unique_masks": 0,
            "mask_entropy": 0.0,
            "template_usage": {},
            "template_entropy": 0.0,
        }

    kept = [sum(int(v) for v in mask) for mask in masks]
    mask_keys = ["".join(str(int(v)) for v in mask) for mask in masks]
    usage = dict(Counter(template_ids or mask_keys))
    return {
        "average_kept_layers": sum(kept) / len(kept),
        "unique_masks": len(set(mask_keys)),
        "mask_entropy": mask_entropy(masks),
        "template_usage": usage,
        "template_entropy": assignment_entropy(list(template_ids or mask_keys)),
    }


def select_template_ids_from_scores(scores, templates: Sequence[LayerTemplate]) -> Tuple[List[str], List[List[int]]]:
    """Select the highest scoring finite template for each row of router scores.

    `scores` is intentionally duck-typed to avoid a hard dependency on torch here.
    It must support `.detach().float().cpu()` and matrix multiplication in callers.
    """
    import torch

    score_tensor = scores.detach().float().cpu()
    template_tensor = torch.tensor([template.mask for template in templates], dtype=score_tensor.dtype)
    utilities = score_tensor @ template_tensor.t()
    selected = torch.argmax(utilities, dim=-1).tolist()
    selected_templates = [templates[int(idx)] for idx in selected]
    return [template.template_id for template in selected_templates], [template.mask for template in selected_templates]
