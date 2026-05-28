import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .mask_library import LayerMaskSpec, generate_mask, mask_id_from_mask
from .mask_utils import keep_count_from_skip_rate


DEFAULT_CANDIDATE_STRATEGIES = (
    "uniform",
    "ends_heavy",
    "shortgpt",
    "first_k",
    "last_k",
    "middle_heavy",
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_risk_label_rows(path: str) -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    if not path:
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["sample_id"])] = row
    return rows


def average_risk_mask(
    risk_rows: Dict[int, Dict[str, object]],
    num_layers: int,
    keep_count: int,
) -> Optional[List[int]]:
    if not risk_rows:
        return None
    totals = [0.0] * int(num_layers)
    counts = [0] * int(num_layers)
    for row in risk_rows.values():
        labels = [float(v) for v in row.get("risk_labels", [])]
        if len(labels) != int(num_layers):
            continue
        for idx, value in enumerate(labels):
            totals[idx] += value
            counts[idx] += 1
    if not all(counts):
        return None
    avg = [totals[idx] / counts[idx] for idx in range(int(num_layers))]
    skip_count = max(0, int(num_layers) - int(keep_count))
    skip_layers = {idx for idx, _ in sorted(enumerate(avg), key=lambda item: (item[1], item[0]))[:skip_count]}
    return [0 if idx in skip_layers else 1 for idx in range(int(num_layers))]


def _append_unique(
    specs: List[LayerMaskSpec],
    seen: set,
    mask: Sequence[int],
    strategy: str,
    mask_id: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> bool:
    row = tuple(int(v) for v in mask)
    if row in seen:
        return False
    seen.add(row)
    specs.append(
        LayerMaskSpec(
            mask_id=mask_id or mask_id_from_mask(row, prefix=strategy),
            strategy=strategy,
            budget=sum(row),
            num_layers=len(row),
            mask=list(row),
            metadata=dict(metadata or {}),
        )
    )
    return True


def candidate_mask_specs(
    num_layers: int,
    top_k_layers: int,
    seed: int,
    candidate_count: int = 16,
    risk_label_file: str = "",
    strategies: Sequence[str] = DEFAULT_CANDIDATE_STRATEGIES,
) -> List[LayerMaskSpec]:
    specs: List[LayerMaskSpec] = []
    seen = set()
    candidate_count = max(1, int(candidate_count))
    top_k_layers = max(0, min(int(top_k_layers), int(num_layers)))

    for strategy in strategies:
        if len(specs) >= candidate_count:
            break
        mask = generate_mask(strategy, num_layers, top_k_layers, seed=seed)
        _append_unique(
            specs,
            seen,
            mask,
            strategy,
            mask_id=f"{strategy}_k{sum(mask)}",
            metadata={"source": "position_proxy"},
        )

    risk_rows = load_risk_label_rows(risk_label_file)
    avg_mask = average_risk_mask(risk_rows, num_layers, top_k_layers)
    if avg_mask is not None and len(specs) < candidate_count:
        _append_unique(
            specs,
            seen,
            avg_mask,
            "average_risk_greedy",
            mask_id=f"average_risk_greedy_k{sum(avg_mask)}",
            metadata={"source": "one_layer_drop_average", "risk_label_file": risk_label_file},
        )

    variant = 0
    while len(specs) < candidate_count:
        mask = generate_mask("random_diverse", num_layers, top_k_layers, seed=seed, variant=variant)
        _append_unique(
            specs,
            seen,
            mask,
            "random_diverse",
            mask_id=f"random_diverse_k{sum(mask)}_v{variant}",
            metadata={"source": "random_same_budget", "seed": int(seed), "variant": int(variant)},
        )
        variant += 1
        if variant > candidate_count * 50:
            break

    return specs[:candidate_count]


def candidate_specs_to_dicts(specs: Sequence[LayerMaskSpec]) -> List[Dict[str, object]]:
    return [asdict(spec) for spec in specs]


def candidate_specs_from_metadata(rows: Iterable[Dict[str, object]]) -> List[LayerMaskSpec]:
    specs = []
    for row in rows:
        mask = [int(v) for v in row["mask"]]
        specs.append(
            LayerMaskSpec(
                mask_id=str(row.get("mask_id") or mask_id_from_mask(mask)),
                strategy=str(row.get("strategy") or "unknown"),
                budget=int(row.get("budget") or sum(mask)),
                num_layers=int(row.get("num_layers") or len(mask)),
                mask=mask,
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return specs


def keep_count_from_args(num_layers: int, skip_rate: float, top_k_layers: int) -> int:
    return keep_count_from_skip_rate(
        int(num_layers),
        float(skip_rate),
        int(top_k_layers) if int(top_k_layers) > 0 else int(num_layers),
    )


def write_json(path: str, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deterministic_sample_indices(
    total: int,
    max_samples: int,
    seed: int,
    strategy: str,
    allowed: Optional[Iterable[int]] = None,
) -> List[int]:
    indices = sorted(int(idx) for idx in (allowed if allowed is not None else range(int(total))) if 0 <= int(idx) < total)
    if max_samples <= 0 or max_samples >= len(indices):
        return indices
    if strategy == "random":
        rng = random.Random(int(seed))
        return sorted(rng.sample(indices, int(max_samples)))
    return indices[: int(max_samples)]
