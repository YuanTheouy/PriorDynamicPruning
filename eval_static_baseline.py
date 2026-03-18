import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import json
import math
import numpy as np
from accelerate import Accelerator

from models.pruned_teacher import PrunedTeacherWrapper
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    
    pad_token_id = 0
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def get_static_mask(strategy, num_layers, top_k):
    """
    Generate a static binary mask based on strategy.
    Returns: torch.Tensor of shape [num_layers] (0.0 or 1.0)
    """
    mask = torch.zeros(num_layers)
    
    if strategy == "uniform":
        # Uniformly sample indices
        indices = np.linspace(0, num_layers - 1, top_k, dtype=int)
        mask[indices] = 1.0
        
    elif strategy == "first_k":
        # Keep first k layers
        mask[:top_k] = 1.0
        
    elif strategy == "last_k":
        # Keep last k layers
        mask[-top_k:] = 1.0
        
    elif strategy == "ends_heavy":
        # Keep start and end layers, prune middle
        # Split top_k into half-half
        k_start = top_k // 2
        k_end = top_k - k_start
        mask[:k_start] = 1.0
        mask[-k_end:] = 1.0
        
    elif strategy == "middle_heavy":
        # Keep middle layers
        start_idx = (num_layers - top_k) // 2
        mask[start_idx : start_idx + top_k] = 1.0
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="./results/static_baseline_result.json")
    parser.add_argument("--top_k_layers", type=int, default=12, help="Number of layers to keep")
    parser.add_argument("--top_k_items", type=int, default=50, help="Number of items to retrieve")
    parser.add_argument("--strategy", type=str, required=True, choices=["uniform", "first_k", "last_k", "ends_heavy", "middle_heavy"])
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}, Strategy: {args.strategy}, Top-K Layers: {args.top_k_layers}")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    
    # 2. Dataset
    dataset = EvalSidDataset(args.test_file, tokenizer, max_len=1024, test=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # 3. Load Teacher
    if accelerator.is_main_process:
        print("Loading Teacher model...")
    from models.modeling_qwen2 import Qwen2ForCausalLM
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.eval()
    
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        num_layers = 24
    
    pruned_teacher = PrunedTeacherWrapper(raw_teacher)
    pruned_teacher.to(device)

    # 4. Construct Static Mask
    static_mask_cpu = get_static_mask(args.strategy, num_layers, args.top_k_layers)
    # Ensure static mask has correct dtype (BFloat16) to match model
    static_mask = static_mask_cpu.to(dtype=torch.bfloat16).to(device).unsqueeze(0) # [1, num_layers]
    
    if accelerator.is_main_process:
        print(f"Static Mask: {static_mask_cpu.tolist()}")
        print(f"Active Layers: {torch.nonzero(static_mask_cpu).squeeze().tolist()}")

    # Trie Logic
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

    # 5. Inference Loop
    all_predictions = []
    
    if accelerator.is_main_process:
        print("Starting Static Baseline Inference...")
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
        
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_size = input_ids.size(0)
            
            # Use fixed mask for all samples in batch
            mask = static_mask.repeat(batch_size, 1) # [batch, num_layers]
            
            # --- Autoregressive Generation with Flattened Batching ---
            batch_candidates = [[{"tokens": [], "score": 0.0, "past_key_values": None}] for _ in range(batch_size)]
            
            for step in range(3):
                flat_input_ids = []
                flat_attention_mask = []
                flat_layer_mask = []
                flat_past_key_values = []
                flat_map = [] 
                
                for b in range(batch_size):
                    for c_idx, cand in enumerate(batch_candidates[b]):
                        curr_tokens = cand["tokens"]
                        past_kv = cand["past_key_values"]
                        
                        if step == 0:
                            # Step 0: Input the full prompt
                            curr_input = input_ids[b]
                            curr_mask = attention_mask[b]
                        else:
                            # Step > 0: Only input the newly generated token
                            # The length of input is 1, and attention mask adds 1
                            curr_input = torch.tensor([curr_tokens[-1]], device=device)
                            # Attention mask needs to cover the past + 1
                            past_seq_len = past_kv[0][0].shape[2] if past_kv is not None else attention_mask[b].shape[0]
                            curr_mask = torch.cat([attention_mask[b], torch.ones(step, device=device)], dim=0)
                            
                        flat_input_ids.append(curr_input)
                        flat_attention_mask.append(curr_mask)
                        flat_layer_mask.append(mask[b])
                        flat_past_key_values.append(past_kv)
                        flat_map.append((b, c_idx))
                
                if not flat_input_ids: continue

                # Pad sequences (Only needed in step 0, in step > 0 all inputs are length 1)
                flat_input_ids = torch.nn.utils.rnn.pad_sequence(flat_input_ids, batch_first=True, padding_value=pad_token_id)
                flat_attention_mask = torch.nn.utils.rnn.pad_sequence(flat_attention_mask, batch_first=True, padding_value=0)
                flat_layer_mask = torch.stack(flat_layer_mask)
                
                # Assemble past_key_values for the batch if step > 0
                batched_past_key_values = None
                if step > 0:
                    # Qwen2 past_key_values is a tuple of tuples: ( (k_layer0, v_layer0), (k_layer1, v_layer1), ... )
                    num_layers_kv = len(flat_past_key_values[0])
                    batched_past_key_values = []
                    for l in range(num_layers_kv):
                        k_list = [kv[l][0] for kv in flat_past_key_values]
                        v_list = [kv[l][1] for kv in flat_past_key_values]
                        batched_past_key_values.append((torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)))
                    batched_past_key_values = tuple(batched_past_key_values)
                
                mini_batch_size = 16 
                num_items = flat_input_ids.size(0)
                all_next_logits = []
                all_new_past_kvs = []
                
                for i in range(0, num_items, mini_batch_size):
                    mb_input = flat_input_ids[i:i+mini_batch_size]
                    mb_mask = flat_attention_mask[i:i+mini_batch_size]
                    mb_layer_mask = flat_layer_mask[i:i+mini_batch_size]
                    
                    mb_past_kv = None
                    if batched_past_key_values is not None:
                        mb_past_kv = tuple(
                            (k[i:i+mini_batch_size], v[i:i+mini_batch_size]) 
                            for (k, v) in batched_past_key_values
                        )
                    
                    logits, new_past_kv = pruned_teacher(
                        mb_input, 
                        mb_mask, 
                        mb_layer_mask, 
                        past_key_values=mb_past_kv, 
                        use_cache=True
                    )
                    next_logits = logits[:, -1, :] 
                    all_next_logits.append(next_logits)
                    
                    # Unpack mini-batch past_key_values into individual samples
                    for j in range(mb_input.size(0)):
                        sample_kv = tuple((k[j:j+1], v[j:j+1]) for k, v in new_past_kv)
                        all_new_past_kvs.append(sample_kv)
                    
                all_next_logits = torch.cat(all_next_logits, dim=0)
                
                new_batch_candidates = [[] for _ in range(batch_size)]
                
                for i, (b, c_idx) in enumerate(flat_map):
                    logits = all_next_logits[i]
                    cand = batch_candidates[b][c_idx]
                    curr_tokens = cand["tokens"]
                    curr_score = cand["score"]
                    new_kv = all_new_past_kvs[i]
                    
                    if step == 0:
                        hash_key = prefix_key
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
                    
                    topk_vals, topk_indices = torch.topk(log_probs, args.top_k_items)
                    
                    for k in range(args.top_k_items):
                        val = topk_vals[k].item()
                        idx = topk_indices[k].item()
                        if val == float('-inf'): continue
                        
                        new_batch_candidates[b].append({
                            "tokens": curr_tokens + [idx],
                            "score": curr_score + val,
                            "past_key_values": new_kv
                        })
                
                for b in range(batch_size):
                    new_batch_candidates[b].sort(key=lambda x: x["score"], reverse=True)
                    batch_candidates[b] = new_batch_candidates[b][:args.top_k_items]
            
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

    rank_output_file = args.output_file.replace(".json", f"_rank{accelerator.process_index}.json")
    os.makedirs(os.path.dirname(rank_output_file), exist_ok=True)
    with open(rank_output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"[Rank {accelerator.process_index}] Saved {len(all_predictions)} predictions to {rank_output_file}")
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
