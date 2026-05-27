import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# [CRITICAL] Inject local transformers library dynamically based on script location
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

import transformers
print(f"DEBUG: Transformers library path: {transformers.__file__}")

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from models.one_layer_student import OneLayerStudentModel
from models.router import LayerRouter, mask_distillation_loss, risk_ranking_loss
from models.pruned_teacher import PrunedTeacherWrapper
from utils_distill import get_sid_token_ids_from_info
from data import SidSFTDataset
from opal_llm.oracle_masks import load_oracle_cache


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[index])
        row["index"] = index
        return row

def collate_fn(batch):
    # Filter out None values that might come from dataset
    batch = [b for b in batch if b is not None]
    
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0) # Update pad value based on tokenizer later
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def set_custom_policy(model, mask_payload, action_payload=None):
    model.config.custom_layer_mask = mask_payload
    model.config.custom_layer_actions = action_payload
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = mask_payload
                layer.self_attn.config.custom_layer_actions = action_payload
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def clear_custom_policy(model):
    model.config.custom_layer_mask = None
    model.config.custom_layer_actions = None
    model.config.custom_compensation_config = {"mode": "none", "rank": 0}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config.custom_layer_mask = None
                layer.self_attn.config.custom_layer_actions = None
                layer.self_attn.config.custom_compensation_config = {"mode": "none", "rank": 0}


def gather_last_token_state(last_hidden_state, attention_mask):
    last_indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), last_indices]


def teacher_prefix_hidden_state(model, input_ids, attention_mask, num_layers, prefix_depth):
    prefix_depth = max(0, min(int(prefix_depth), int(num_layers)))
    prefix_mask = [1 if idx < prefix_depth else 0 for idx in range(num_layers)]
    set_custom_policy(
        model,
        mask_payload=prefix_mask,
        action_payload=[1 if keep else 0 for keep in prefix_mask],
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


def oracle_mask_tensor(batch_indices, oracle_cache, num_layers, device):
    if not oracle_cache or batch_indices is None:
        return None
    rows = []
    for index in batch_indices.detach().cpu().tolist():
        entry = oracle_cache.get(int(index))
        if not entry:
            rows.append(None)
            continue
        mask = entry.get("oracle_mask") or entry.get("execution_mask") or entry.get("layer_mask") or []
        if len(mask) != num_layers:
            rows.append(None)
        else:
            rows.append([float(v) for v in mask])
    if any(row is None for row in rows):
        return None
    return torch.tensor(rows, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint")
    parser.add_argument("--student_ckpt", type=str, default="", help="Path to the pretrained student checkpoint")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k_layers", type=int, default=12, help="Number of layers to keep in Teacher")
    parser.add_argument("--output_dir", type=str, default="./policy_ckpts")
    parser.add_argument("--train_student", action="store_true", help="Whether to unfreeze and train the student model")
    parser.add_argument(
        "--router_input_source",
        choices=["student", "teacher_prefix"],
        default="student",
        help="State used by the policy router. Use teacher_prefix for OPAL prefix-hidden training.",
    )
    parser.add_argument("--prefix_depth", type=int, default=4, help="Teacher prefix depth for --router_input_source teacher_prefix")
    parser.add_argument("--oracle_cache", default="", help="Oracle mask cache for OPAL-2 mask/risk distillation.")
    parser.add_argument("--mask_distill_weight", type=float, default=0.0)
    parser.add_argument("--risk_ranking_weight", type=float, default=0.0)
    parser.add_argument("--kl_distill_weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.router_input_source == "student" and not args.student_ckpt:
        raise ValueError("--student_ckpt is required when --router_input_source student")
    if args.router_input_source != "student" and args.train_student:
        raise ValueError("--train_student is only valid with --router_input_source student")

    # Initialize Accelerator
    # Fix unused parameters error in DDP
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}, Total processes: {accelerator.num_processes}")
        if args.router_input_source == "teacher_prefix":
            print(f"Training Strategy: Teacher Prefix Hidden State + Policy Router (prefix_depth={args.prefix_depth})")
        elif args.train_student:
            print("🚀 Training Strategy: Jointly training Student Encoder + Policy Router")
        else:
            print("🧊 Training Strategy: Frozen Student Encoder, Training Policy Router Only")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    if accelerator.is_main_process:
        print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    oracle_cache = load_oracle_cache(args.oracle_cache) if args.oracle_cache else {}
    if args.oracle_cache and not oracle_cache:
        raise ValueError(f"--oracle_cache has no usable oracle rows: {args.oracle_cache}")
    
    # 2. Dataset & DataLoader
    dataset = SidSFTDataset(
        train_file=args.train_file, 
        tokenizer=tokenizer, 
        category=args.category,
        max_len=1024
    )
    
    def custom_collate(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        indices = [int(b.get("index", -1)) for b in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if any(index >= 0 for index in indices):
            result["index"] = torch.tensor(indices, dtype=torch.long)
        return result

    if oracle_cache:
        dataset = IndexedDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)

    # 3. Load Teacher
    if accelerator.is_main_process:
        print("Loading Teacher model...")
    # Load raw teacher
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.eval()
    
    # Wrap teacher for pruning
    # Note: PrunedTeacherWrapper keeps raw_teacher internally
    pruned_teacher = PrunedTeacherWrapper(raw_teacher)
    pruned_teacher.to(device)
    
    # Identify number of layers
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        # Fallback or error
        num_layers = 24 # Assumption
        if accelerator.is_main_process:
            print(f"Warning: Could not detect number of layers, assuming {num_layers}")

    if accelerator.is_main_process:
        print(f"Teacher has {num_layers} layers. Target Top-K: {args.top_k_layers}")

    # 4. Load the router state source.
    student = None
    if args.router_input_source == "student":
        if accelerator.is_main_process:
            print("Loading Student model...")
        student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
        
        checkpoint = torch.load(args.student_ckpt, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            student.load_state_dict(checkpoint['model_state_dict'])
        else:
            student.load_state_dict(checkpoint)

        student.to(torch.bfloat16)
        student.to(device)

        if args.train_student:
            student.train()
            for param in student.backbone.parameters():
                param.requires_grad = True
        else:
            student.eval()
            for param in student.parameters():
                param.requires_grad = False
    elif accelerator.is_main_process:
        print(f"Training OPAL router from teacher prefix hidden states at depth {args.prefix_depth}.")

    # 5. Initialize Router (Policy Network)
    hidden_size = student.backbone.config.hidden_size if student is not None else raw_teacher.config.hidden_size
    router = LayerRouter(hidden_size=hidden_size, num_layers=num_layers, top_k=args.top_k_layers)
    router.to(torch.bfloat16) # Match precision
    router.to(device)
    router.train()

    # Optimizer
    # If training student, include its parameters
    if args.train_student:
        params_to_optimize = list(router.parameters()) + list(student.parameters())
    else:
        params_to_optimize = list(router.parameters())
        
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)

    # Prepare with Accelerate
    if args.router_input_source == "student" and args.train_student:
        router, student, optimizer, dataloader = accelerator.prepare(router, student, optimizer, dataloader)
    else:
        router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    # 6. Training Loop
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
            
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            batch_indices = batch.get("index")

            # Filter valid positions
            valid_positions_mask = (labels != -100)
            if not valid_positions_mask.any():
                continue

            # --- 1. Get Router State ---
            if args.router_input_source == "student":
                if args.train_student:
                    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    with torch.no_grad():
                        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                state = gather_last_token_state(student_outputs["last_hidden_state"], attention_mask)
            else:
                prefix_hidden = teacher_prefix_hidden_state(
                    raw_teacher,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_layers=num_layers,
                    prefix_depth=args.prefix_depth,
                )
                state = gather_last_token_state(prefix_hidden, attention_mask)

            # --- 2. Router Forward ---
            # mask: [batch, num_layers]
            mask, scores = router(state)

            # --- 3. Full Teacher Forward (Target) ---
            with torch.no_grad():
                # We can call raw_teacher directly
                full_outputs = raw_teacher(input_ids=input_ids, attention_mask=attention_mask)
                full_logits = full_outputs.logits
                
                # Shift and filter for SID
                shift_full_logits = full_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_valid_mask = (shift_labels != -100)
                
                active_full_logits = shift_full_logits[shift_valid_mask]
                active_full_sid_logits = active_full_logits[:, sid_token_ids]
                
                # Target Probs
                target_probs = F.softmax(active_full_sid_logits / args.temperature, dim=-1)

            # --- 4. Pruned Teacher Forward (Prediction) ---
            # We use the wrapper. It requires layer_mask.
            pruned_logits = pruned_teacher(input_ids, attention_mask, mask)
            
            # Shift and filter
            shift_pruned_logits = pruned_logits[..., :-1, :].contiguous()
            # Re-use mask and indices
            active_pruned_logits = shift_pruned_logits[shift_valid_mask]
            active_pruned_sid_logits = active_pruned_logits[:, sid_token_ids]
            
            # Log Probs for KL
            log_pruned_probs = F.log_softmax(active_pruned_sid_logits / args.temperature, dim=-1)

            # --- 5. Debug Metrics (Approx NDCG/HR) ---
            if accelerator.is_main_process and total_loss == 0: # Only print once per epoch start to avoid flooding
                with torch.no_grad():
                    # Get top predictions for both
                    top_full = torch.argmax(active_full_sid_logits, dim=-1) # [num_active]
                    top_pruned = torch.argmax(active_pruned_sid_logits, dim=-1) # [num_active]
                    
                    # Match rate between pruned and full teacher
                    match_rate = (top_full == top_pruned).float().mean()
                    
                    # Target item match rate
                    # active_labels are the token_ids we want to predict
                    active_labels = shift_labels[shift_valid_mask]
                    # We need to map active_labels to sid_token_ids indices
                    # This is complex, let's just use token_ids directly for a quick check
                    full_preds_ids = torch.tensor(sid_token_ids, device=device)[top_full]
                    pruned_preds_ids = torch.tensor(sid_token_ids, device=device)[top_pruned]
                    
                    full_acc = (full_preds_ids == active_labels).float().mean()
                    pruned_acc = (pruned_preds_ids == active_labels).float().mean()
                    
                    print(f"\nDEBUG: [Match with Full: {match_rate:.4f}] [Full Acc: {full_acc:.4f}] [Pruned Acc: {pruned_acc:.4f}]")
                    print(f"DEBUG: Mask sample (first 5 layers): {mask[0, :5].tolist()}")

            # --- 6. Loss ---
            kl_loss = F.kl_div(log_pruned_probs, target_probs, reduction='batchmean') * (args.temperature ** 2)
            loss = args.kl_distill_weight * kl_loss
            oracle_mask = oracle_mask_tensor(batch_indices, oracle_cache, num_layers, device)
            if oracle_mask is not None and args.mask_distill_weight > 0:
                loss = loss + args.mask_distill_weight * mask_distillation_loss(scores, oracle_mask)
            if oracle_mask is not None and args.risk_ranking_weight > 0:
                loss = loss + args.risk_ranking_weight * risk_ranking_loss(scores, oracle_mask)

            # Backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
            
            # Save Checkpoint
            save_path = os.path.join(args.output_dir, f"policy_epoch_{epoch+1}.pt")
            unwrapped_router = accelerator.unwrap_model(router)
            
            # Save Policy
            router_metadata = {
                "router_input_source": args.router_input_source,
                "prefix_depth": int(args.prefix_depth) if args.router_input_source == "teacher_prefix" else 0,
                "top_k_layers": int(args.top_k_layers),
                "num_layers": int(num_layers),
                "teacher_model": args.teacher_model,
                "train_student": bool(args.train_student),
                "mask_training_mode": "differentiable_soft_mask",
                "oracle_cache": args.oracle_cache,
                "oracle_cache_size": len(oracle_cache),
                "mask_distill_weight": float(args.mask_distill_weight),
                "risk_ranking_weight": float(args.risk_ranking_weight),
                "kl_distill_weight": float(args.kl_distill_weight),
            }
            torch.save(
                {
                    "model_state_dict": unwrapped_router.state_dict(),
                    "metadata": router_metadata,
                },
                save_path,
            )
            print(f"Saved policy checkpoint to {save_path}")
            
            # If we trained student, we should save it too!
            if args.train_student:
                student_save_path = os.path.join(args.output_dir, f"finetuned_student_epoch_{epoch+1}.pt")
                unwrapped_student = accelerator.unwrap_model(student)
                torch.save({"model_state_dict": unwrapped_student.state_dict()}, student_save_path)
                print(f"Saved fine-tuned student to {student_save_path}")

if __name__ == "__main__":
    main()
