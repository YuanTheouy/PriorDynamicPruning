from typing import Dict, List, Optional, Sequence


FORMAL_ORACLE_METHODS = {"single_drop", "greedy"}
FORMAL_ORACLE_REQUIRED_FIELDS = {
    "sample_id",
    "skip_rate",
    "num_layers",
    "oracle_method",
    "oracle_mask",
    "oracle_loss",
    "full_loss",
    "best_static_loss",
    "candidate_count",
    "oracle_regret_reference",
}


def _mask_key(mask: Sequence[int]) -> str:
    return "".join(str(int(v)) for v in mask)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    width = min(len(left), len(right))
    return sum(1 for idx in range(width) if int(left[idx]) != int(right[idx])) + abs(len(left) - len(right))


def load_oracle_cache(path: str) -> Dict[int, Dict[str, object]]:
    if not path:
        return {}
    import json

    payload = json.loads(open(path, encoding="utf-8").read())
    rows = payload.get("oracle", payload.get("predictions", []))
    cache = {}
    for row in rows:
        sample_index = row.get("index", row.get("sample_id"))
        if sample_index is None:
            continue
        oracle_mask = row.get("oracle_mask") or row.get("execution_mask") or row.get("layer_mask") or []
        if not oracle_mask:
            continue
        normalized = dict(row)
        normalized["index"] = int(sample_index)
        normalized["sample_id"] = int(sample_index)
        normalized["oracle_mask"] = [int(v) for v in oracle_mask]
        normalized["oracle_mask_key"] = _mask_key(normalized["oracle_mask"])
        cache[int(sample_index)] = normalized
    return cache


def validate_formal_oracle_cache(cache: Dict[int, Dict[str, object]], expected_method: str) -> None:
    if expected_method not in FORMAL_ORACLE_METHODS:
        raise ValueError(f"Expected a formal oracle method, got {expected_method!r}.")
    for sample_index, entry in cache.items():
        method = str(entry.get("oracle_method") or "")
        objective = entry.get("objective", entry.get("oracle_objective"))
        missing = [
            field
            for field in FORMAL_ORACLE_REQUIRED_FIELDS
            if field not in entry or entry.get(field) is None
        ]
        if objective is None:
            missing.append("objective")
        if method != expected_method:
            raise ValueError(
                "Oracle cache method mismatch for sample "
                f"{sample_index}: expected {expected_method!r}, found {method!r}."
            )
        if missing:
            raise ValueError(
                "Formal OPAL-2 oracle cache row is missing required fields for sample "
                f"{sample_index}: {sorted(set(missing))}"
            )


def candidate_for_mask(oracle_entry: Dict[str, object], mask: Sequence[int]) -> Optional[Dict[str, object]]:
    target_key = _mask_key(mask)
    for candidate in oracle_entry.get("oracle_candidates", []) or []:
        candidate_mask = candidate.get("oracle_mask") or candidate.get("execution_mask") or candidate.get("layer_mask") or []
        if candidate_mask and _mask_key(candidate_mask) == target_key:
            return candidate
    return None


def oracle_eval_fields(oracle_entry: Optional[Dict[str, object]], predicted_mask: Sequence[int]) -> Dict[str, object]:
    if not oracle_entry:
        return {
            "oracle_mask": [],
            "oracle_loss": None,
            "full_loss": None,
            "mask_regret": None,
            "prefix_to_oracle_agreement": None,
            "hamming_distance": None,
            "oracle_candidates": [],
        }

    oracle_mask = [int(v) for v in oracle_entry.get("oracle_mask", [])]
    oracle_loss = oracle_entry.get("oracle_loss", oracle_entry.get("NLL_skip"))
    full_loss = oracle_entry.get("full_loss", oracle_entry.get("NLL_full"))
    oracle_objective = oracle_entry.get("oracle_objective", oracle_entry.get("objective"))
    candidate = candidate_for_mask(oracle_entry, predicted_mask)
    distance = hamming_distance(oracle_mask, predicted_mask)
    predicted_loss = None
    if candidate is not None:
        predicted_loss = candidate.get("oracle_loss", candidate.get("NLL_skip", candidate.get("loss")))
    if predicted_loss is not None and oracle_loss is not None:
        mask_regret = float(predicted_loss) - float(oracle_loss)
    elif candidate is not None and candidate.get("rank") is not None and oracle_entry.get("oracle_rank") is not None:
        mask_regret = float(candidate["rank"]) - float(oracle_entry["oracle_rank"])
    elif distance == 0 and oracle_mask:
        mask_regret = 0.0
    else:
        mask_regret = None

    return {
        "oracle_mask": oracle_mask,
        "oracle_loss": oracle_loss,
        "full_loss": full_loss,
        "oracle_method": oracle_entry.get("oracle_method"),
        "oracle_objective": oracle_objective,
        "best_static_loss": oracle_entry.get("best_static_loss"),
        "candidate_count": oracle_entry.get("candidate_count"),
        "oracle_regret_reference": oracle_entry.get("oracle_regret_reference"),
        "mask_regret": mask_regret,
        "prefix_to_oracle_agreement": 1.0 if distance == 0 and oracle_mask else 0.0,
        "hamming_distance": distance,
        "oracle_candidates": oracle_entry.get("oracle_candidates", []) or [],
    }
