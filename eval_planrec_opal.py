#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
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
from opal_llm.template_library import (
    DEFAULT_TEMPLATE_STRATEGIES,
    LayerTemplate,
    filter_templates,
    generate_mask,
    generate_template_library,
    load_template_library,
    save_template_library,
    select_template_ids_from_scores,
)
from opal_llm.timing import GenerationTimer

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


def load_or_build_templates(args, num_layers: int) -> List[LayerTemplate]:
    if args.template_library:
        templates = load_template_library(args.template_library)
    else:
        budgets = parse_int_csv(args.budgets) if args.budgets else [args.top_k_layers]
        templates = generate_template_library(
            num_layers=num_layers,
            budgets=budgets,
            strategies=parse_strategy_csv(args.template_strategies),
            random_templates_per_budget=args.random_templates_per_budget,
            seed=args.seed,
        )
        if args.save_template_library:
            save_template_library(args.save_template_library, templates)
    return templates


def static_template_from_args(args, num_layers: int, templates: Sequence[LayerTemplate]) -> LayerTemplate:
    if args.template_id:
        return filter_templates(templates, template_id=args.template_id)[0]
    if args.static_strategy:
        mask = generate_mask(args.static_strategy, num_layers, args.top_k_layers, seed=args.seed)
        return LayerTemplate(
            template_id=f"{args.static_strategy}_k{sum(mask)}",
            strategy=args.static_strategy,
            budget=sum(mask),
            num_layers=num_layers,
            mask=mask,
            metadata={"source": "static_strategy"},
        )
    return filter_templates(templates, budget=args.top_k_layers)[0]


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
    student.load_state_dict(student_ckpt.get("model_state_dict", student_ckpt))
    student.to(dtype_from_precision(args.precision)).to(device).eval()

    hidden_size = student.backbone.config.hidden_size
    router = LayerRouter(hidden_size=hidden_size, num_layers=num_layers, top_k=args.top_k_layers)
    router_ckpt = torch.load(args.policy_ckpt, map_location="cpu")
    router.load_state_dict(router_ckpt.get("model_state_dict", router_ckpt))
    router.to(dtype_from_precision(args.precision)).to(device).eval()
    return student, router


def dynamic_masks(
    args,
    student,
    router,
    input_ids,
    attention_mask,
    templates: Sequence[LayerTemplate],
    num_layers: int,
):
    with torch.no_grad():
        student_out = student(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = student_out["last_hidden_state"]
        last_indices = attention_mask.sum(dim=1) - 1
        state = last_hidden_state[torch.arange(input_ids.size(0), device=input_ids.device), last_indices]
        mask, scores = router(state)

    if args.method == "dynamic" and args.dynamic_selection == "template_library":
        budget_templates = filter_templates(templates, budget=args.top_k_layers)
        template_ids, masks = select_template_ids_from_scores(scores, budget_templates)
        return masks, template_ids, scores.detach().float().cpu().tolist()

    hard_masks = [[int(round(float(v))) for v in row] for row in mask.detach().float().cpu().tolist()]
    template_ids = [f"router_topk_k{sum(row)}" for row in hard_masks]
    return hard_masks, template_ids, scores.detach().float().cpu().tolist()


def input_guided_masks(args, input_ids, attention_mask, templates: Sequence[LayerTemplate]):
    """Select finite templates from prompt/input features, without executing an LLM prefix.

    This is a lightweight PuDDing/IG-style baseline hook. It intentionally uses the
    same template library as OPAL but only cheap prompt features, so any gain from
    OPAL can be attributed to internal prefix evidence rather than to the library.
    """
    budget_templates = filter_templates(templates, budget=args.top_k_layers)
    features = input_guided_features(input_ids, attention_mask, args.input_guided_num_bins)

    template_ids = []
    masks = []
    row_features = []
    calibrated_map = getattr(args, "_input_guided_calibrated_map", {}) or {}
    template_by_id = {template.template_id: template for template in budget_templates}
    for feature in features:
        if args.input_guided_selector == "length":
            selector_value = int(feature["length"])
        elif args.input_guided_selector == "hash":
            selector_value = int(feature["token_sum"])
        elif args.input_guided_selector == "calibrated":
            key = input_guided_feature_key(feature, args.input_guided_feature_set)
            template_id = calibrated_map.get(key)
            template = template_by_id.get(template_id)
            if template is None:
                selector_value = int(feature["selector_value"])
                template = budget_templates[(selector_value + int(args.seed)) % len(budget_templates)]
            template_ids.append(template.template_id)
            masks.append(template.mask)
            row_features.append(summarize_input_guided_feature(feature, args.input_guided_feature_set))
            continue
        else:
            selector_value = int(feature["selector_value"])
        idx = (selector_value + int(args.seed)) % len(budget_templates)
        template = budget_templates[idx]
        template_ids.append(template.template_id)
        masks.append(template.mask)
        row_features.append(summarize_input_guided_feature(feature, args.input_guided_feature_set))
    return masks, template_ids, row_features


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


def load_input_guided_calibration(args) -> Dict[str, str]:
    if not args.input_guided_calibration_json:
        if args.input_guided_selector == "calibrated":
            raise ValueError("--input_guided_calibration_json is required for calibrated input-guided selection")
        return {}
    payload = json.loads(open(args.input_guided_calibration_json, encoding="utf-8").read())
    rows = payload.get("predictions", [])
    votes: Dict[str, Counter] = defaultdict(Counter)
    rank_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    rank_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        feature = row.get("input_guided_features") or row.get("input_features") or {}
        if not feature:
            continue
        key = str(feature.get("feature_key") or input_guided_feature_key(feature, args.input_guided_feature_set))
        oracle_template = str(row.get("template_id", ""))
        if oracle_template:
            votes[key][oracle_template] += 1
        for candidate in row.get("oracle_candidates", []) or []:
            template_id = str(candidate.get("template_id", ""))
            if not template_id:
                continue
            rank_sums[(key, template_id)] += float(candidate.get("rank", 1_000_000))
            rank_counts[(key, template_id)] += 1

    calibrated = {}
    keys = set(votes)
    keys.update(key for key, _ in rank_sums)
    for key in keys:
        scored = [
            (rank_sums[(key, template_id)] / rank_counts[(key, template_id)], template_id)
            for row_key, template_id in rank_counts
            if row_key == key and rank_counts[(key, template_id)] > 0
        ]
        if scored:
            calibrated[key] = min(scored)[1]
        elif votes.get(key):
            calibrated[key] = votes[key].most_common(1)[0][0]
    if args.input_guided_selector == "calibrated" and not calibrated:
        raise ValueError(f"No calibrated input-guided entries found in {args.input_guided_calibration_json}")
    return calibrated


def set_custom_mask(model, mask_payload):
    model.config.custom_layer_mask = mask_payload
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = mask_payload


def clear_custom_mask(model):
    set_custom_mask(model, None)


def generate_once(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
    mask_payload,
    timer: GenerationTimer,
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
        if mask_payload is None:
            clear_custom_mask(model)
        else:
            set_custom_mask(model, mask_payload)
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
            clear_custom_mask(model)

    output = timer.measure(call_generate, samples=input_ids.size(0))
    decoded = decode_generation(tokenizer, args.teacher_model, output.sequences, input_ids.size(1))
    return group_beam_outputs(decoded, input_ids.size(0), args.top_k_items)


def generate_grouped_by_template(
    model,
    tokenizer,
    args,
    input_ids,
    attention_mask,
    prefix_allowed_tokens_fn,
    masks: Sequence[Sequence[int]],
    timer: GenerationTimer,
):
    groups = defaultdict(list)
    for idx, mask in enumerate(masks):
        groups[tuple(int(v) for v in mask)].append(idx)

    grouped_outputs = [None] * len(masks)
    for mask_tuple, row_indices in groups.items():
        idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=input_ids.device)
        sub_outputs = generate_once(
            model=model,
            tokenizer=tokenizer,
            args=args,
            input_ids=input_ids.index_select(0, idx_tensor),
            attention_mask=attention_mask.index_select(0, idx_tensor),
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            mask_payload=list(mask_tuple),
            timer=timer,
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
    templates: Sequence[LayerTemplate],
    batch_indices: Sequence[int],
    ground_truths: Sequence[str],
    timer: GenerationTimer,
):
    best = {
        int(index): {"rank": 1_000_000, "predictions": [], "template": None, "candidates": []}
        for index in batch_indices
    }
    for template in filter_templates(templates, budget=args.top_k_layers):
        outputs = generate_once(
            model=model,
            tokenizer=tokenizer,
            args=args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            mask_payload=template.mask,
            timer=timer,
        )
        for row_pos, sample_index in enumerate(batch_indices):
            target = ground_truths[int(sample_index)] if int(sample_index) < len(ground_truths) else ""
            rank = rank_of(outputs[row_pos], target)
            candidate = {"template_id": template.template_id, "rank": rank}
            if args.save_oracle_candidates:
                best[int(sample_index)]["candidates"].append(candidate)
            if rank < best[int(sample_index)]["rank"]:
                best[int(sample_index)].update(
                    {
                        "rank": rank,
                        "predictions": outputs[row_pos],
                        "template": template,
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
        choices=["full", "static", "dynamic", "input_guided", "layerwise_router", "oracle"],
        required=True,
    )
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--student_ckpt", default="")
    parser.add_argument("--policy_ckpt", default="")
    parser.add_argument("--template_library", default="")
    parser.add_argument("--save_template_library", default="")
    parser.add_argument("--template_id", default="")
    parser.add_argument("--template_strategies", default=",".join(DEFAULT_TEMPLATE_STRATEGIES))
    parser.add_argument("--static_strategy", default="uniform")
    parser.add_argument("--dynamic_selection", choices=["topk_mask", "template_library"], default="template_library")
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
    parser.add_argument("--random_templates_per_budget", type=int, default=0)
    parser.add_argument("--group_by_template", action="store_true")
    parser.add_argument("--save_oracle_candidates", action="store_true")
    parser.add_argument("--drop_dedup", action="store_true")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", default="./results/planrec_experiments")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
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
    effective_top_k_layers = num_layers if args.method == "full" else args.top_k_layers

    from utils_distill import get_sid_token_ids_from_info

    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    templates = load_or_build_templates(args, num_layers)
    args._input_guided_calibrated_map = load_input_guided_calibration(args)
    static_template = None
    if args.method == "static":
        static_template = static_template_from_args(args, num_layers, templates)

    student = router = None
    if args.method in {"dynamic", "layerwise_router"}:
        student, router = load_student_router(args, sid_token_ids, num_layers, device)

    timer = GenerationTimer(warmup_batches=args.warmup_batches, timed_batches=args.timed_batches)
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
                    "num_templates": len(templates),
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

        if args.method == "full":
            masks = [[1] * num_layers for _ in range(batch_size)]
            template_ids = ["full_teacher"] * batch_size
            outputs = generate_once(
                model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, None, timer
            )
        elif args.method == "static":
            masks = [static_template.mask for _ in range(batch_size)]
            template_ids = [static_template.template_id for _ in range(batch_size)]
            outputs = generate_once(
                model,
                tokenizer,
                args,
                input_ids,
                attention_mask,
                prefix_allowed_tokens_fn,
                static_template.mask,
                timer,
            )
        elif args.method == "dynamic":
            masks, template_ids, router_scores = dynamic_masks(
                args, student, router, input_ids, attention_mask, templates, num_layers
            )
            if args.group_by_template:
                outputs = generate_grouped_by_template(
                    model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, timer
                )
            else:
                outputs = generate_once(
                    model,
                    tokenizer,
                    args,
                    input_ids,
                    attention_mask,
                    prefix_allowed_tokens_fn,
                    masks,
                    timer,
                )
        else:
            if args.method == "input_guided":
                masks, template_ids, input_guided_features = input_guided_masks(
                    args, input_ids, attention_mask, templates
                )
                if args.group_by_template:
                    outputs = generate_grouped_by_template(
                        model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, timer
                    )
                else:
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        timer,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    predictions.append(
                        {
                            "index": int(sample_index),
                            "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                            "sample_predictions": outputs[local_pos],
                            "layer_mask": masks[local_pos],
                            "template_id": template_ids[local_pos],
                            "input_guided_features": input_guided_features[local_pos],
                        }
                    )
                continue

            if args.method == "layerwise_router":
                masks, template_ids, router_scores = dynamic_masks(
                    args, student, router, input_ids, attention_mask, templates, num_layers
                )
                if args.group_by_template:
                    outputs = generate_grouped_by_template(
                        model, tokenizer, args, input_ids, attention_mask, prefix_allowed_tokens_fn, masks, timer
                    )
                else:
                    outputs = generate_once(
                        model,
                        tokenizer,
                        args,
                        input_ids,
                        attention_mask,
                        prefix_allowed_tokens_fn,
                        masks,
                        timer,
                    )
                for local_pos, sample_index in enumerate(batch_indices):
                    predictions.append(
                        {
                            "index": int(sample_index),
                            "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                            "sample_predictions": outputs[local_pos],
                            "layer_mask": masks[local_pos],
                            "template_id": template_ids[local_pos],
                            "router_scores": router_scores[local_pos],
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
                templates,
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
                template = row["template"]
                predictions.append(
                    {
                        "index": int(sample_index),
                        "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                        "sample_predictions": row["predictions"],
                        "layer_mask": template.mask if template else [],
                        "template_id": template.template_id if template else "none",
                        "oracle_rank": row["rank"],
                        "oracle_candidates": row["candidates"] if args.save_oracle_candidates else [],
                        "input_guided_features": summarize_input_guided_feature(
                            feature, args.input_guided_feature_set
                        ),
                    }
                )
            continue

        for local_pos, sample_index in enumerate(batch_indices):
            record = {
                "index": int(sample_index),
                "input": tokenizer.decode(input_ids[local_pos], skip_special_tokens=True),
                "sample_predictions": outputs[local_pos],
                "layer_mask": masks[local_pos],
                "template_id": template_ids[local_pos],
            }
            if args.method == "dynamic":
                record["router_scores"] = router_scores[local_pos]
            predictions.append(record)

    rank_path = dirs["raw_json"] / f"{run_name}_rank{accelerator.process_index}.json"
    write_json(
        rank_path,
        {
            "predictions": predictions,
            "timing_records": timer.raw_records(),
        },
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_predictions = []
        timing_records = []
        for path in sorted(dirs["raw_json"].glob(f"{run_name}_rank*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_predictions.extend(payload.get("predictions", []))
            timing_records.extend(payload.get("timing_records", []))
        all_predictions = sorted(all_predictions, key=lambda row: int(row.get("index", 0)))

        metadata = {
            "dataset": args.category,
            "method": args.method,
            "top_k_layers": effective_top_k_layers,
            "checkpoint": args.teacher_model,
            "student_ckpt": args.student_ckpt,
            "policy_ckpt": args.policy_ckpt,
            "template_library": args.template_library or args.save_template_library,
            "template_id": args.template_id,
            "static_strategy": args.static_strategy,
            "dynamic_selection": args.dynamic_selection,
            "input_guided_selector": args.input_guided_selector,
            "input_guided_calibration_json": args.input_guided_calibration_json,
            "input_guided_feature_set": args.input_guided_feature_set,
            "input_guided_num_bins": args.input_guided_num_bins,
            "input_guided_calibrated_keys": len(args._input_guided_calibrated_map),
            "group_by_template": args.group_by_template,
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
            }
        else:
            latency_summary = timer.summary()

        final_payload = build_payload(
            metadata=metadata,
            predictions=all_predictions,
            ground_truths=ground_truths,
            valid_sids=valid_sids,
            latency=latency_summary,
            timing_records=timing_records,
        )
        final_path = dirs["raw_json"] / f"{run_name}.json"
        write_json(final_path, final_payload)
        append_summary_csv(dirs["tables"] / "summary.csv", summary_row(final_payload, str(final_path)))
        write_json(dirs["logs"] / f"{run_name}_metadata.json", metadata)
        print(f"Saved final OPAL result JSON to {final_path}")
        print(f"Updated summary table at {dirs['tables'] / 'summary.csv'}")


if __name__ == "__main__":
    main()
