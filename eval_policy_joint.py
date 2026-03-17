import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import json
import math
from accelerate import Accelerator

from models.one_layer_student import OneLayerStudentModel
from models.router import LayerRouter
from models.pruned_teacher import PrunedTeacherWrapper
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def collate_fn(batch):
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    # EvalSidDataset returns {"input_ids": tokens, "attention_mask": mask} for test=True
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    
    # Pad sequences
    # Note: Tokenizer pad_token_id should be available. 
    # If not passed in args, we assume 0 or handle it.
    # We can get pad_token_id from the tokenizer used in main, but here it's global?
    # Better to pass tokenizer or use a default.
    pad_token_id = 0 # Default fallback
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint")
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to the trained student checkpoint (.pt file)")
    parser.add_argument("--policy_ckpt", type=str, required=True, help="Path to the trained policy router checkpoint (.pt file)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="./results/policy_joint_result.json")
    parser.add_argument("--top_k_layers", type=int, default=12, help="Number of layers to keep in Teacher")
    parser.add_argument("--top_k_items", type=int, default=50, help="Number of items to retrieve")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}, Total processes: {accelerator.num_processes}")

    # 1. Prepare Tokenizer & SID Vocab
    # Use simple from_pretrained which loads fast
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Update collate_fn's pad token
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    if accelerator.is_main_process:
        print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    
    local2global_sid = {local_idx: global_id for local_idx, global_id in enumerate(sid_token_ids)}
    global2local_sid = {global_id: local_idx for local_idx, global_id in enumerate(sid_token_ids)}
    
    # Load Item Info & Trie Constraints (Same as eval_student.py)
    item_dict = {}
    with open(args.info_file, 'r') as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip() for _ in items]
        
    info_semantic = [f'''### Response:\n{_}\n''' for _ in item_names]
    # Simple check for model type (assuming Qwen/Llama)
    if args.teacher_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        
    # Heuristic for prefix length
    if args.teacher_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
        
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
        
    first_ID = prefixID[0]
    prefix_key = get_hash(first_ID[:prefix_index])
    valid_start_globals = hash_dict.get(prefix_key, [])
    valid_start_locals = [global2local_sid[g] for g in valid_start_globals if g in global2local_sid]
    if accelerator.is_main_process:
        print(f"Constraining Step 1 to {len(valid_start_locals)} valid start tokens (Key: {prefix_key})")

    # 2. Load Models
    if accelerator.is_main_process:
        print("Loading Models...")
    
    # A. Student (Feature Extractor)
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    ckpt = torch.load(args.student_ckpt, map_location='cpu')
    if 'model_state_dict' in ckpt:
        student.load_state_dict(ckpt['model_state_dict'])
    else:
        student.load_state_dict(ckpt)
    student.to(torch.bfloat16).to(device).eval()
    
    # B. Teacher (Base for Pruning)
    raw_teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.eval()
    
    # Detect num_layers
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        num_layers = 24
    if accelerator.is_main_process:
        print(f"Teacher has {num_layers} layers.")

    pruned_teacher = PrunedTeacherWrapper(raw_teacher)
    pruned_teacher.to(device)
    
    # C. Router (Policy)
    hidden_size = student.backbone.config.hidden_size
    router = LayerRouter(hidden_size, num_layers, args.top_k_layers)
    router_ckpt = torch.load(args.policy_ckpt, map_location='cpu')
    # Handle if saved as state_dict or full dict
    if isinstance(router_ckpt, dict) and 'model_state_dict' in router_ckpt:
        router.load_state_dict(router_ckpt['model_state_dict'])
    else:
        router.load_state_dict(router_ckpt)
    router.to(torch.bfloat16).to(device).eval()

    # 3. Dataset
    # Pass test=True to get only input_ids and attention_mask without labels
    dataset = EvalSidDataset(args.test_file, tokenizer, max_len=1024, test=True)
    
    # Use custom collate_fn defined above
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Prepare DataLoader with Accelerate (Splits data across GPUs)
    dataloader = accelerator.prepare(dataloader)

    # 4. Evaluation Loop
    all_predictions = []
    
    if accelerator.is_main_process:
        print("Starting Joint Inference...")
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
        
    with torch.no_grad():
        for batch in pbar:
            # Data is already on correct device via accelerate
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # --- Step 0: Get Router Mask ---
            # Forward Student to get state
            student_out = student(input_ids, attention_mask)
            last_hidden_state = student_out["last_hidden_state"]
            # Get last non-pad token state
            last_indices = attention_mask.sum(dim=1) - 1
            state = last_hidden_state[torch.arange(input_ids.size(0), device=device), last_indices]
            
            # Forward Router
            # mask: [batch, num_layers]
            mask, _ = router(state)
            
            # --- Autoregressive Generation (3 Steps) with Beam Search/Top-K ---
            # To match eval_student.py logic for accurate NDCG, we need to generate Top-K candidates.
            # Simplified Logic: 
            # Step 1: Get Top-K start tokens
            # Step 2: Expand each to Top-K next tokens (filtered by Trie)
            # Step 3: Expand each to Top-K next tokens (filtered by Trie)
            # Finally keep Top-K sequence candidates
            
            # Note: Implementing full beam search inside this loop for pruned teacher is complex.
            # We use a simplified beam search where we keep top_k candidates at each step.
            
            batch_preds = []
            
            for b in range(input_ids.size(0)):
                item_input_ids = input_ids[b:b+1]
                item_att_mask = attention_mask[b:b+1]
                item_layer_mask = mask[b:b+1]
                
                # Candidates: list of dicts {"tokens": [id1, id2...], "score": log_prob}
                candidates = [{"tokens": [], "score": 0.0}]
                
                # --- Token 1 ---
                logits = pruned_teacher(item_input_ids, item_att_mask, item_layer_mask)
                next_token_logits = logits[0, -1, :] # [vocab]
                
                # Apply SID constraint (Step 1)
                full_vocab_mask = torch.full_like(next_token_logits, float('-inf'))
                if len(valid_start_globals) > 0:
                    valid_ids_tensor = torch.tensor(valid_start_globals, device=device)
                    full_vocab_mask[valid_ids_tensor] = 0
                
                next_token_logits = next_token_logits + full_vocab_mask
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Top K
                topk_vals, topk_indices = torch.topk(log_probs, args.top_k_items)
                
                new_candidates = []
                for k in range(args.top_k_items):
                    val = topk_vals[k].item()
                    idx = topk_indices[k].item()
                    if val == float('-inf'): continue
                    new_candidates.append({"tokens": [idx], "score": val})
                
                candidates = new_candidates
                
                # --- Token 2 ---
                # Expand candidates
                expanded_candidates = []
                for cand in candidates:
                    curr_tokens = cand["tokens"]
                    curr_score = cand["score"]
                    
                    # Prepare input
                    temp_input = torch.cat([item_input_ids, torch.tensor([curr_tokens], device=device)], dim=1)
                    temp_mask = torch.cat([item_att_mask, torch.tensor([[1]*len(curr_tokens)], device=device)], dim=1)
                    
                    logits = pruned_teacher(temp_input, temp_mask, item_layer_mask)
                    next_token_logits = logits[0, -1, :]
                    
                    # Constraint (Trie)
                    # Get allowed next tokens
                    hash_key = get_hash(curr_tokens) # prefix_key + hash? No, hash_dict stores hash of prefix.
                    # Wait, hash_dict keys are hashes of PREFIXES.
                    # Token 1 is a prefix for Token 2.
                    # In eval_student: hash_key = get_hash(cand_tokens)
                    # Here cand_tokens is [token1].
                    # We need to ensure we use the same hashing logic.
                    # hash_dict construction:
                    # for i in range(prefix_index, len(ID)): ... hash_dict[hash_number].add(ID[i])
                    # If i=prefix_index, hash_number = get_hash(ID[:i]) -> this is the prefix prompt hash (prefix_key).
                    # If i=prefix_index+1 (Token 2), hash_number = get_hash(ID[prefix_index:i]) -> hash([token1])
                    
                    # So for Token 2, key is hash([token1]).
                    # But wait, `get_hash` joins with '-'.
                    # We need to make sure we hash ONLY the new tokens, not the prompt.
                    # eval_student logic: `hash_key = get_hash(cand_tokens)` where cand_tokens are the generated SID tokens.
                    # Yes, that matches.
                    
                    hash_key = get_hash(curr_tokens)
                    allowed_globals = hash_dict.get(hash_key, [])
                    allowed_locals = [global2local_sid[g] for g in allowed_globals if g in global2local_sid]
                    # Wait, allowed_globals are GLOBAL ids. pruned_teacher outputs GLOBAL logits.
                    # We don't need local conversion unless we were using Student's SID head.
                    # Teacher outputs full vocab. So we need global ids.
                    
                    mask_trie = torch.full_like(next_token_logits, float('-inf'))
                    if allowed_globals:
                         # Filter allowed_globals to be within vocab range
                         valid_allowed = [g for g in allowed_globals if g < next_token_logits.size(0)]
                         if valid_allowed:
                             mask_trie[torch.tensor(valid_allowed, device=device)] = 0
                    
                    next_token_logits = next_token_logits + mask_trie
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # We want global Top-K across all expansions? Or Top-K per branch?
                    # Standard beam search: Top-K per branch then global Top-K?
                    # Simplified: Top-K per branch (width=1 or small) -> Here we iterate all candidates.
                    # Let's take Top-K from this branch.
                    topk_vals, topk_indices = torch.topk(log_probs, args.top_k_items) # Keep top-k to allow diversity
                    
                    for k in range(args.top_k_items):
                        val = topk_vals[k].item()
                        idx = topk_indices[k].item()
                        if val == float('-inf'): continue
                        expanded_candidates.append({
                            "tokens": curr_tokens + [idx],
                            "score": curr_score + val
                        })
                
                # Prune to Top-K globally
                expanded_candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = expanded_candidates[:args.top_k_items]
                
                # --- Token 3 ---
                final_candidates = []
                for cand in candidates:
                    curr_tokens = cand["tokens"]
                    curr_score = cand["score"]
                    
                    temp_input = torch.cat([item_input_ids, torch.tensor([curr_tokens], device=device)], dim=1)
                    temp_mask = torch.cat([item_att_mask, torch.tensor([[1]*len(curr_tokens)], device=device)], dim=1)
                    
                    logits = pruned_teacher(temp_input, temp_mask, item_layer_mask)
                    next_token_logits = logits[0, -1, :]
                    
                    hash_key = get_hash(curr_tokens)
                    allowed_globals = hash_dict.get(hash_key, [])
                    
                    mask_trie = torch.full_like(next_token_logits, float('-inf'))
                    if allowed_globals:
                         valid_allowed = [g for g in allowed_globals if g < next_token_logits.size(0)]
                         if valid_allowed:
                             mask_trie[torch.tensor(valid_allowed, device=device)] = 0
                    
                    next_token_logits = next_token_logits + mask_trie
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    topk_vals, topk_indices = torch.topk(log_probs, args.top_k_items)
                    
                    for k in range(args.top_k_items):
                        val = topk_vals[k].item()
                        idx = topk_indices[k].item()
                        if val == float('-inf'): continue
                        final_candidates.append({
                            "tokens": curr_tokens + [idx],
                            "score": curr_score + val
                        })
                        
                final_candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = final_candidates[:args.top_k_items]
                
                # Save Top-K predictions strings
                sample_preds = []
                for cand in candidates:
                    pred_tokens = tokenizer.convert_ids_to_tokens(cand["tokens"])
                    pred_str = "".join(pred_tokens).replace('Ġ', '')
                    sample_preds.append(pred_str)
                
                batch_preds.append({
                    "input": tokenizer.decode(input_ids[b], skip_special_tokens=True),
                    "sample_predictions": sample_preds, # Save all candidates for NDCG
                    "layer_mask": item_layer_mask.cpu().tolist()
                })
            
            all_predictions.extend(batch_preds)

    # Save Partial Results per Rank
    # We use rank specific filename to avoid conflicts
    rank_output_file = args.output_file.replace(".json", f"_rank{accelerator.process_index}.json")
    
    os.makedirs(os.path.dirname(rank_output_file), exist_ok=True)
    with open(rank_output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"[Rank {accelerator.process_index}] Saved {len(all_predictions)} predictions to {rank_output_file}")
    
    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
