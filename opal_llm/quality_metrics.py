import math
from typing import Dict, List, Sequence, Tuple


def _finite_exp(value: float) -> float:
    if not math.isfinite(value):
        return float("inf")
    return float(math.exp(min(80.0, value)))


def summarize_quality_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "NLL_full",
        "NLL_skip",
        "Delta_NLL",
        "PPL_full",
        "PPL_skip",
        "Delta_PPL",
        "KL_full_to_skip",
        "oracle_regret",
    ]
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        summary[key] = sum(values) / len(values) if values else 0.0
    summary["num_quality_samples"] = len(rows)
    return summary


def encode_target_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, bos=False, eos=True)
    except TypeError:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids = list(token_ids) + [int(eos_token_id)]
        return token_ids


def build_target_scoring_batch(tokenizer, input_ids, attention_mask, targets: Sequence[str], max_len: int):
    import torch

    rows = []
    label_rows = []
    for ids, mask, target in zip(input_ids.detach().cpu(), attention_mask.detach().cpu(), targets):
        prompt_ids = ids[mask.bool()].tolist()
        target_ids = encode_target_ids(tokenizer, f"{str(target).strip()}\n")
        combined = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if max_len > 0 and len(combined) > max_len:
            combined = combined[-max_len:]
            labels = labels[-max_len:]
        rows.append(combined)
        label_rows.append(labels)

    pad_token_id = tokenizer.pad_token_id
    width = max(len(row) for row in rows)
    padded_ids = []
    padded_masks = []
    padded_labels = []
    for row, labels in zip(rows, label_rows):
        pad = width - len(row)
        padded_ids.append([pad_token_id] * pad + row)
        padded_masks.append([0] * pad + [1] * len(row))
        padded_labels.append([-100] * pad + labels)

    device = input_ids.device
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(padded_masks, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


def quality_rows_from_logits(full_logits, skip_logits, labels) -> List[Dict[str, float]]:
    import torch
    import torch.nn.functional as F

    full_shift = full_logits[..., :-1, :].float().contiguous()
    skip_shift = skip_logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)

    full_log_probs = F.log_softmax(full_shift, dim=-1)
    skip_log_probs = F.log_softmax(skip_shift, dim=-1)
    full_probs = full_log_probs.exp()
    token_kl = (full_probs * (full_log_probs - skip_log_probs)).sum(dim=-1)

    safe_labels = shift_labels.clamp(min=0)
    full_target_nll = -full_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    skip_target_nll = -skip_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    rows = []
    for idx in range(labels.size(0)):
        row_valid = valid[idx]
        count = int(row_valid.sum().item())
        if count <= 0:
            rows.append(
                {
                    "NLL_full": 0.0,
                    "NLL_skip": 0.0,
                    "Delta_NLL": 0.0,
                    "PPL_full": 1.0,
                    "PPL_skip": 1.0,
                    "Delta_PPL": 0.0,
                    "KL_full_to_skip": 0.0,
                    "quality_token_count": 0,
                }
            )
            continue
        nll_full = float(full_target_nll[idx][row_valid].mean().item())
        nll_skip = float(skip_target_nll[idx][row_valid].mean().item())
        kl_value = float(token_kl[idx][row_valid].mean().item())
        ppl_full = _finite_exp(nll_full)
        ppl_skip = _finite_exp(nll_skip)
        rows.append(
            {
                "NLL_full": nll_full,
                "NLL_skip": nll_skip,
                "Delta_NLL": nll_skip - nll_full,
                "PPL_full": ppl_full,
                "PPL_skip": ppl_skip,
                "Delta_PPL": ppl_skip - ppl_full,
                "KL_full_to_skip": kl_value,
                "quality_token_count": count,
            }
        )
    return rows
