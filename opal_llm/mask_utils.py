from typing import Dict, List, Optional, Sequence, Tuple


def keep_count_from_skip_rate(num_layers: int, skip_rate: float, fallback_top_k: int) -> int:
    num_layers = int(num_layers)
    if num_layers <= 0:
        return 0
    if skip_rate is None or float(skip_rate) < 0:
        return max(0, min(num_layers, int(fallback_top_k)))
    skip_count = int(round(num_layers * float(skip_rate)))
    skip_count = max(0, min(num_layers, skip_count))
    return num_layers - skip_count


def actual_skip_rate(mask: Sequence[int]) -> float:
    if not mask:
        return 0.0
    return 1.0 - (sum(int(v) for v in mask) / float(len(mask)))


def mask_from_skip_risk(skip_risk: Sequence[float], keep_count: int) -> List[int]:
    keep_count = max(0, min(len(skip_risk), int(keep_count)))
    ranked = sorted(range(len(skip_risk)), key=lambda idx: float(skip_risk[idx]), reverse=True)
    keep = set(ranked[:keep_count])
    return [1 if idx in keep else 0 for idx in range(len(skip_risk))]


def allowed_layers_from_protected(
    num_layers: int,
    protected_head: Optional[int] = None,
    protected_tail: Optional[int] = None,
) -> List[int]:
    protected_head = max(0, int(protected_head or 0))
    protected_tail = max(0, int(protected_tail or 0))
    end = max(protected_head, int(num_layers) - protected_tail)
    return list(range(protected_head, end))


def mask_from_skip_risk_allowed(
    skip_risk: Sequence[float],
    keep_count: int,
    allowed_layers: Optional[Sequence[int]] = None,
) -> List[int]:
    if allowed_layers is None:
        return mask_from_skip_risk(skip_risk, keep_count)
    num_layers = len(skip_risk)
    keep_count = max(0, min(num_layers, int(keep_count)))
    skip_count = num_layers - keep_count
    allowed = [int(idx) for idx in allowed_layers if 0 <= int(idx) < num_layers]
    if skip_count > len(allowed):
        raise ValueError(f"skip_count={skip_count} exceeds allowed layer count={len(allowed)}")
    mask = [1 for _ in range(num_layers)]
    skipped = sorted(allowed, key=lambda idx: float(skip_risk[idx]))[:skip_count]
    for idx in skipped:
        mask[idx] = 0
    return mask


def max_consecutive_skips(mask: Sequence[int]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if int(value) == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def stage_bounds(num_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    num_stages = max(1, int(num_stages))
    bounds = []
    for stage in range(num_stages):
        start = (stage * num_layers) // num_stages
        end = ((stage + 1) * num_layers) // num_stages
        if start < end:
            bounds.append((start, end))
    return bounds


def stage_keep_counts(mask: Sequence[int], num_stages: int) -> List[int]:
    return [sum(int(v) for v in mask[start:end]) for start, end in stage_bounds(len(mask), num_stages)]


def structure_penalty_value(
    mask: Sequence[int],
    max_consecutive: int,
    num_stages: int,
    min_keep_per_stage: int,
) -> float:
    penalty = 0.0
    if max_consecutive > 0:
        penalty += max(0, max_consecutive_skips(mask) - int(max_consecutive))
    if min_keep_per_stage > 0:
        for count in stage_keep_counts(mask, num_stages):
            penalty += max(0, int(min_keep_per_stage) - int(count))
    return float(penalty)


def _swap_keep_for_skip(mask: List[int], skip_risk: Sequence[float], force_keep_idx: int) -> bool:
    if mask[force_keep_idx] == 1:
        return True
    removable = [idx for idx, value in enumerate(mask) if int(value) == 1 and idx != force_keep_idx]
    if not removable:
        return False
    remove_idx = min(removable, key=lambda idx: float(skip_risk[idx]))
    mask[force_keep_idx] = 1
    mask[remove_idx] = 0
    return True


def repair_structure_constraints(
    base_mask: Sequence[int],
    skip_risk: Sequence[float],
    max_consecutive: int = 0,
    num_stages: int = 1,
    min_keep_per_stage: int = 0,
    max_passes: int = 8,
) -> Tuple[List[int], Dict[str, object]]:
    mask = [int(v) for v in base_mask]
    if not mask:
        return mask, {
            "max_consecutive_skips_actual": 0,
            "stage_keep_counts": [],
            "structure_penalty_value": 0.0,
        }

    for _ in range(max(1, int(max_passes))):
        changed = False
        if min_keep_per_stage > 0:
            for start, end in stage_bounds(len(mask), num_stages):
                while sum(mask[start:end]) < int(min_keep_per_stage):
                    skipped = [idx for idx in range(start, end) if mask[idx] == 0]
                    if not skipped:
                        break
                    keep_idx = max(skipped, key=lambda idx: float(skip_risk[idx]))
                    changed = _swap_keep_for_skip(mask, skip_risk, keep_idx) or changed

        if max_consecutive > 0:
            run_start = None
            run_len = 0
            for idx, value in enumerate(mask + [1]):
                if int(value) == 0:
                    if run_start is None:
                        run_start = idx
                    run_len += 1
                    continue
                if run_len > int(max_consecutive) and run_start is not None:
                    run_indices = list(range(run_start, run_start + run_len))
                    needed = run_len - int(max_consecutive)
                    for keep_idx in sorted(run_indices, key=lambda pos: float(skip_risk[pos]), reverse=True)[:needed]:
                        changed = _swap_keep_for_skip(mask, skip_risk, keep_idx) or changed
                run_start = None
                run_len = 0

        if not changed:
            break

    stats = {
        "max_consecutive_skips_actual": max_consecutive_skips(mask),
        "stage_keep_counts": stage_keep_counts(mask, num_stages),
        "structure_penalty_value": structure_penalty_value(
            mask,
            max_consecutive=max_consecutive,
            num_stages=num_stages,
            min_keep_per_stage=min_keep_per_stage,
        ),
    }
    return mask, stats
