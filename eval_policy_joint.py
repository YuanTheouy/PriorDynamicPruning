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
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    if accelerator.is_main_process:
        print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    
    local2global_sid = {local_idx: global_id for local_idx, global_id in enumerate(sid_token_ids)}
    global2local_sid = {global_id: local_idx for local_idx, global_id in enumerate(sid_token_ids)}
    
    item_dict = {}
    with open(args.info_file, 'r') as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip() for _ in items]
        
    info_semantic = [f'''### Response:\n{_}\n''' for _ in item_names]
    if args.teacher_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        
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
    
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    ckpt = torch.load(args.student_ckpt, map_location='cpu')
    if 'model_state_dict' in ckpt:
        student.load_state_dict(ckpt['model_state_dict'])
    else:
        student.load_state_dict(ckpt)
    student.to(torch.bfloat16).to(device).eval()
    
    from models.modeling_qwen2 import Qwen2ForCausalLM
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.eval()
    
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
    
    hidden_size = student.backbone.config.hidden_size
    router = LayerRouter(hidden_size, num_layers, args.top_k_layers)
    router_ckpt = torch.load(args.policy_ckpt, map_location='cpu')
    if isinstance(router_ckpt, dict) and 'model_state_dict' in router_ckpt:
        router.load_state_dict(router_ckpt['model_state_dict'])
    else:
        router.load_state_dict(router_ckpt)
    router.to(torch.bfloat16).to(device).eval()

    # 3. Dataset
    dataset = EvalSidDataset(args.test_file, tokenizer, max_len=1024, test=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
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
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_size = input_ids.size(0)
            
            # --- Step 0: Get Router Mask ---
            student_out = student(input_ids, attention_mask)
            last_hidden_state = student_out["last_hidden_state"]
            last_indices = attention_mask.sum(dim=1) - 1
            state = last_hidden_state[torch.arange(batch_size, device=device), last_indices]
            mask, _ = router(state) # [batch, num_layers]
            
            # --- Autoregressive Generation with Flattened Batching ---
            
            # Current Candidates: List of Top-K candidates for each sample in batch
            # Structure: batch_candidates[b] = list of {"tokens": [...], "score": ...}
            batch_candidates = [[{"tokens": [], "score": 0.0}] for _ in range(batch_size)]
            
            # We will perform 3 generation steps
            for step in range(3):
                # Flatten all candidates to run a single large batch
                flat_input_ids = []
                flat_attention_mask = []
                flat_layer_mask = []
                
                # Metadata to reconstruct batch structure
                # Mapping from flat_idx -> (batch_idx, candidate_idx)
                flat_map = [] 
                
                for b in range(batch_size):
                    for c_idx, cand in enumerate(batch_candidates[b]):
                        curr_tokens = cand["tokens"]
                        # Construct input: prompt + generated tokens
                        if len(curr_tokens) > 0:
                            curr_input = torch.cat([input_ids[b], torch.tensor(curr_tokens, device=device)])
                            curr_mask = torch.cat([attention_mask[b], torch.tensor([1]*len(curr_tokens), device=device)])
                        else:
                            curr_input = input_ids[b]
                            curr_mask = attention_mask[b]
                            
                        flat_input_ids.append(curr_input)
                        flat_attention_mask.append(curr_mask)
                        flat_layer_mask.append(mask[b])
                        flat_map.append((b, c_idx))
                
                # Pad flattened batch
                flat_input_ids = torch.nn.utils.rnn.pad_sequence(flat_input_ids, batch_first=True, padding_value=pad_token_id)
                flat_attention_mask = torch.nn.utils.rnn.pad_sequence(flat_attention_mask, batch_first=True, padding_value=0)
                flat_layer_mask = torch.stack(flat_layer_mask)
                
                # Split into mini-batches to avoid OOM
                # Teacher model is large, so we process e.g. 16 at a time
                mini_batch_size = 16 
                num_items = flat_input_ids.size(0)
                all_next_logits = []
                
                for i in range(0, num_items, mini_batch_size):
                    mb_input = flat_input_ids[i:i+mini_batch_size]
                    mb_mask = flat_attention_mask[i:i+mini_batch_size]
                    mb_layer_mask = flat_layer_mask[i:i+mini_batch_size]
                    
                    logits = pruned_teacher(mb_input, mb_mask, mb_layer_mask)
                    # Take logits of the last token
                    next_logits = logits[:, -1, :] # [mini_batch, vocab]
                    all_next_logits.append(next_logits)
                    
                all_next_logits = torch.cat(all_next_logits, dim=0) # [total_candidates, vocab]
                
                # Process logits and update candidates
                new_batch_candidates = [[] for _ in range(batch_size)]
                
                for i, (b, c_idx) in enumerate(flat_map):
                    logits = all_next_logits[i]
                    cand = batch_candidates[b][c_idx]
                    curr_tokens = cand["tokens"]
                    curr_score = cand["score"]
                    
                    # Constraint
                    if step == 0:
                        hash_key = prefix_key
                        # For step 0, we use prefix_key
                        allowed_globals = hash_dict.get(hash_key, [])
                    else:
                        hash_key = get_hash(curr_tokens)
                        allowed_globals = hash_dict.get(hash_key, [])
                        
                    mask_trie = torch.full_like(logits, float('-inf'))
                    if allowed_globals:
                         valid_allowed = [g for g in allowed_globals if g < logits.size(0)]
                         if valid_allowed:
                             mask_trie[torch.tensor(valid_allowed, device=device)] = 0
                    
                    logits = logits + mask_trie
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Top-K expansion per candidate
                    topk_vals, topk_indices = torch.topk(log_probs, args.top_k_items)
                    
                    for k in range(args.top_k_items):
                        val = topk_vals[k].item()
                        idx = topk_indices[k].item()
                        if val == float('-inf'): continue
                        
                        new_batch_candidates[b].append({
                            "tokens": curr_tokens + [idx],
                            "score": curr_score + val
                        })
                
                # Prune to Top-K per batch item
                for b in range(batch_size):
                    new_batch_candidates[b].sort(key=lambda x: x["score"], reverse=True)
                    batch_candidates[b] = new_batch_candidates[b][:args.top_k_items]
            
            # Finalize Batch Predictions
            for b in range(batch_size):
                sample_preds = []
                for cand in batch_candidates[b]:
                    pred_tokens = tokenizer.convert_ids_to_tokens(cand["tokens"])
                    pred_str = "".join(pred_tokens).replace('Ġ', '')
                    sample_preds.append(pred_str)
                
                all_predictions.append({
                    "input": tokenizer.decode(input_ids[b], skip_special_tokens=True),
                    "sample_predictions": sample_preds,
                    "layer_mask": mask[b].cpu().tolist()
                })

    # Save Partial Results
    rank_output_file = args.output_file.replace(".json", f"_rank{accelerator.process_index}.json")
    os.makedirs(os.path.dirname(rank_output_file), exist_ok=True)
    with open(rank_output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"[Rank {accelerator.process_index}] Saved {len(all_predictions)} predictions to {rank_output_file}")
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
