import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from opal_llm.mask_utils import keep_count_from_skip_rate


C6_STATIC_STRATEGIES = (
    "uniform",
    "ends_heavy",
    "first_k",
    "last_k",
    "middle_heavy",
    "random_diverse_seed42",
)
RELATED_CANDIDATE_STRATEGIES = (
    "uniform",
    "ends_heavy",
    "first_k",
    "last_k",
    "middle_heavy",
    "random_diverse_seed42",
    "random_diverse_v1",
    "random_diverse_v2",
    "random_diverse_v3",
    "random_diverse_v4",
    "random_diverse_v5",
    "random_diverse_v6",
    "random_diverse_v7",
    "random_diverse_v8",
    "random_diverse_v9",
    "random_diverse_v10",
)


def dtype_from_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def finite_exp(value: float) -> float:
    if not math.isfinite(value):
        return float("inf")
    return float(math.exp(min(80.0, float(value))))


def detect_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Could not detect decoder layer count.")


def set_custom_policy(model, mask_payload) -> None:
    model.config.custom_layer_mask = mask_payload
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    layer_owner = getattr(model, "model", model)
    if hasattr(layer_owner, "layers"):
        for layer in layer_owner.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = mask_payload
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def clear_custom_policy(model) -> None:
    model.config.custom_layer_mask = None
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    layer_owner = getattr(model, "model", model)
    if hasattr(layer_owner, "layers"):
        for layer in layer_owner.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = None
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def resolve_skip_budget(
    num_layers: int,
    skip_rate: float,
    skip_count: int = 0,
) -> Tuple[int, int]:
    num_layers = int(num_layers)
    if int(skip_count) > 0:
        k_skip = max(0, min(num_layers, int(skip_count)))
        return k_skip, num_layers - k_skip
    keep_count = keep_count_from_skip_rate(num_layers, float(skip_rate), num_layers)
    k_skip = num_layers - int(keep_count)
    return k_skip, int(keep_count)


def allowed_layers_from_policy(
    num_layers: int,
    protected_head: int,
    protected_tail: int,
) -> List[int]:
    protected_head = max(0, int(protected_head))
    protected_tail = max(0, int(protected_tail))
    end = max(protected_head, int(num_layers) - protected_tail)
    return list(range(protected_head, end))


def protected_layers_from_allowed(num_layers: int, allowed_layers: Sequence[int]) -> List[int]:
    allowed = {int(idx) for idx in allowed_layers}
    return [idx for idx in range(int(num_layers)) if idx not in allowed]


def _fill_unique(values: Iterable[int], candidates: Sequence[int], count: int) -> List[int]:
    selected = []
    seen = set()
    candidate_set = {int(idx) for idx in candidates}
    for value in values:
        value = int(value)
        if value in candidate_set and value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= int(count):
            return selected
    for value in candidates:
        value = int(value)
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= int(count):
            break
    return selected


def _select_by_strategy(strategy: str, candidates: Sequence[int], count: int, seed: int) -> List[int]:
    candidates = [int(idx) for idx in candidates]
    count = max(0, min(len(candidates), int(count)))
    if count <= 0:
        return []
    if count >= len(candidates):
        return list(candidates)

    if strategy == "uniform":
        if count == 1:
            positions = [len(candidates) // 2]
        else:
            positions = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
        return _fill_unique((candidates[pos] for pos in positions), candidates, count)
    if strategy == "first_k":
        return list(candidates[:count])
    if strategy == "last_k":
        return list(candidates[-count:])
    if strategy == "ends_heavy":
        left = (count + 1) // 2
        right = count - left
        return _fill_unique(list(candidates[:left]) + list(candidates[-right:] if right else []), candidates, count)
    if strategy == "middle_heavy":
        start = (len(candidates) - count) // 2
        return list(candidates[start : start + count])
    if strategy == "random_diverse_seed42":
        rng = random.Random(42 if seed is None else int(seed))
        return sorted(rng.sample(candidates, count))
    if strategy.startswith("random_diverse_v"):
        variant = int(strategy.rsplit("v", 1)[1])
        rng = random.Random((42 if seed is None else int(seed)) + 1009 * variant)
        return sorted(rng.sample(candidates, count))
    raise ValueError(f"Unknown static strategy: {strategy}")


def static_keep_mask(
    strategy: str,
    num_layers: int,
    skip_count: int,
    protected_head: int,
    protected_tail: int,
    seed: int = 42,
) -> List[int]:
    num_layers = int(num_layers)
    skip_count = max(0, min(num_layers, int(skip_count)))
    keep_count = num_layers - skip_count
    allowed_layers = allowed_layers_from_policy(num_layers, protected_head, protected_tail)
    protected_layers = protected_layers_from_allowed(num_layers, allowed_layers)
    if skip_count > len(allowed_layers):
        raise ValueError(
            f"skip_count={skip_count} exceeds allowed layer count={len(allowed_layers)} "
            f"with protected_head={protected_head}, protected_tail={protected_tail}"
        )
    extra_keep = keep_count - len(protected_layers)
    if extra_keep < 0:
        raise ValueError(
            f"keep_count={keep_count} is smaller than protected layer count={len(protected_layers)}"
        )
    selected_allowed = _select_by_strategy(strategy, allowed_layers, extra_keep, seed=seed)
    keep = set(protected_layers) | set(selected_allowed)
    mask = [1 if idx in keep else 0 for idx in range(num_layers)]
    actual_skip = num_layers - sum(mask)
    if actual_skip != skip_count:
        raise AssertionError(f"Static mask has {actual_skip} skipped layers, expected {skip_count}")
    return mask


def related_candidate_masks(
    num_layers: int,
    skip_count: int,
    protected_head: int,
    protected_tail: int,
    seed: int = 42,
    strategies: Sequence[str] = RELATED_CANDIDATE_STRATEGIES,
) -> List[Dict[str, object]]:
    rows = []
    seen = set()
    for strategy in strategies:
        mask = static_keep_mask(
            strategy,
            num_layers=num_layers,
            skip_count=skip_count,
            protected_head=protected_head,
            protected_tail=protected_tail,
            seed=seed,
        )
        key = mask_key(mask)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "candidate_id": strategy,
                "strategy": strategy,
                "keep_mask": [int(v) for v in mask],
                "skipped_layers": skipped_layers_from_keep_mask(mask),
            }
        )
    return rows


def mask_key(mask: Sequence[int]) -> str:
    return "".join(str(int(v)) for v in mask)


def skipped_layers_from_keep_mask(mask: Sequence[int]) -> List[int]:
    return [idx for idx, value in enumerate(mask) if int(value) == 0]


def keep_mask_from_skipped(
    num_layers: int,
    skipped_layers: Sequence[int],
    batch_size: int,
    device,
) -> torch.Tensor:
    mask = torch.ones((int(batch_size), int(num_layers)), dtype=torch.float32, device=device)
    for layer_idx in skipped_layers:
        mask[:, int(layer_idx)] = 0.0
    return mask


def keep_masks_from_skip_risk(
    risk: torch.Tensor,
    skip_count: int,
    allowed_layers: Sequence[int],
) -> List[List[int]]:
    risk = risk.detach().float().cpu()
    num_layers = int(risk.size(-1))
    skip_count = max(0, min(num_layers, int(skip_count)))
    allowed = [int(idx) for idx in allowed_layers if 0 <= int(idx) < num_layers]
    if skip_count > len(allowed):
        raise ValueError(f"skip_count={skip_count} exceeds allowed layer count={len(allowed)}")
    masks = []
    for row in risk:
        keep_mask = [1 for _ in range(num_layers)]
        skipped = sorted(allowed, key=lambda idx: (float(row[idx].item()), idx))[:skip_count]
        for idx in skipped:
            keep_mask[idx] = 0
        masks.append(keep_mask)
    return masks


def summarize_keep_masks(masks: Sequence[Sequence[int]], expected_skip_count: int) -> Dict[str, object]:
    if not masks:
        return {
            "average_kept_layers": 0.0,
            "average_skipped_layers": 0.0,
            "exact_skip_count_rate": 0.0,
            "unique_masks": 0,
            "mask_usage": {},
        }
    kept = [sum(int(v) for v in mask) for mask in masks]
    skipped = [len(mask) - sum(int(v) for v in mask) for mask in masks]
    keys = [mask_key(mask) for mask in masks]
    return {
        "average_kept_layers": sum(kept) / len(kept),
        "average_skipped_layers": sum(skipped) / len(skipped),
        "exact_skip_count_rate": sum(1 for value in skipped if value == int(expected_skip_count)) / len(skipped),
        "unique_masks": len(set(keys)),
        "mask_usage": dict(Counter(keys).most_common(20)),
    }


def load_wikitext_token_ids(
    tokenizer,
    split: str,
    dataset_disk_path: str = "",
    dataset_cache_dir: str = "",
    dataset_path: str = "wikitext",
    dataset_name: str = "wikitext-2-raw-v1",
) -> List[int]:
    try:
        from datasets import load_dataset, load_from_disk

        if dataset_disk_path:
            disk_dataset = load_from_disk(dataset_disk_path)
            if hasattr(disk_dataset, "keys"):
                if split not in disk_dataset:
                    raise KeyError(
                        f"split={split!r} not found in dataset saved at {dataset_disk_path}; "
                        f"available={list(disk_dataset.keys())}"
                    )
                dataset = disk_dataset[split]
            else:
                dataset = disk_dataset
        else:
            kwargs = {}
            if dataset_cache_dir:
                kwargs["cache_dir"] = dataset_cache_dir
            dataset = load_dataset(dataset_path, dataset_name, split=split, **kwargs)
    except Exception as exc:
        source = f"load_from_disk({dataset_disk_path})" if dataset_disk_path else f"{dataset_path}/{dataset_name}"
        raise RuntimeError(
            "WikiText-2 dataset blocker: could not load "
            f"{source} split={split!r}. "
            "On the server, either point WIKITEXT_DATASET_DISK_PATH at a saved Arrow dataset, "
            "enable HuggingFace access once, or pre-populate the HF cache. "
            f"Original error: {exc}"
        ) from exc
    texts = [str(row.get("text", "")).strip() for row in dataset]
    text = "\n\n".join(part for part in texts if part)
    if not text:
        raise ValueError(f"WikiText-2 split {split!r} is empty after filtering blank rows.")
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [int(value) for value in token_ids]


class WikitextWindowDataset(Dataset):
    def __init__(
        self,
        token_ids: Sequence[int],
        seq_len: int,
        max_windows: int = 0,
        sample_strategy: str = "first",
        sample_seed: int = 42,
        selected_window_ids: Optional[Sequence[int]] = None,
        stride: Optional[int] = None,
    ):
        self.token_ids = [int(value) for value in token_ids]
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if len(self.token_ids) < self.seq_len:
            raise ValueError(
                f"Need at least seq_len={self.seq_len} tokens, got {len(self.token_ids)}"
            )
        self.starts = list(range(0, len(self.token_ids) - self.seq_len + 1, self.stride))
        if not self.starts:
            raise ValueError("No token windows available.")
        if selected_window_ids is not None:
            self.window_ids = [int(idx) for idx in selected_window_ids]
        else:
            self.window_ids = list(range(len(self.starts)))
            if int(max_windows) > 0 and int(max_windows) < len(self.window_ids):
                if sample_strategy == "random":
                    rng = random.Random(int(sample_seed))
                    self.window_ids = sorted(rng.sample(self.window_ids, int(max_windows)))
                elif sample_strategy == "first":
                    self.window_ids = self.window_ids[: int(max_windows)]
                else:
                    raise ValueError(f"Unsupported sample_strategy: {sample_strategy}")
        for window_id in self.window_ids:
            if window_id < 0 or window_id >= len(self.starts):
                raise IndexError(f"selected window id {window_id} is outside [0, {len(self.starts)})")

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        window_id = int(self.window_ids[int(index)])
        start = int(self.starts[window_id])
        ids = self.token_ids[start : start + self.seq_len]
        return {
            "sample_id": window_id,
            "window_start": start,
            "input_ids": ids,
        }


def collate_wikitext_windows(
    batch: Sequence[Dict[str, object]],
    router_prefix_tokens: int,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch")
    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    prefix = max(1, min(int(router_prefix_tokens), input_ids.size(1) - 1))
    labels[:, :prefix] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_id": torch.tensor([int(row["sample_id"]) for row in batch], dtype=torch.long),
        "window_start": torch.tensor([int(row["window_start"]) for row in batch], dtype=torch.long),
    }


def lm_loss_stats_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, float]]:
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.clamp(min=0)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    rows = []
    for idx in range(labels.size(0)):
        row_valid = valid[idx]
        count = int(row_valid.sum().item())
        if count <= 0:
            rows.append({"total_nll": 0.0, "token_count": 0, "nll": 0.0, "ppl": 1.0})
            continue
        total_nll = float(token_loss[idx][row_valid].sum().item())
        nll = total_nll / float(count)
        rows.append(
            {
                "total_nll": total_nll,
                "token_count": count,
                "nll": nll,
                "ppl": finite_exp(nll),
            }
        )
    return rows


def aggregate_loss_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    total_nll = sum(float(row.get("total_nll", 0.0)) for row in rows)
    token_count = sum(int(row.get("token_count", 0)) for row in rows)
    nll = total_nll / float(max(1, token_count))
    return {
        "total_nll": total_nll,
        "eval_tokens": int(token_count),
        "nll": nll,
        "ppl": finite_exp(nll),
    }


def read_jsonl(path: str) -> List[Dict[str, object]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_json(path: str, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
