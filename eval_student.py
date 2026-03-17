import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import json
import math

from models.one_layer_student import OneLayerStudentModel
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint (used for tokenizer and config)")
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to the trained student checkpoint (.pt file)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="./results/student_result.json")
    parser.add_argument("--top_k", type=int, default=50, help="Number of items to retrieve")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" # Crucial for generation/inference
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    
    # Map local index back to global token id
    local2global_sid = {local_idx: global_id for local_idx, global_id in enumerate(sid_token_ids)}
    # And global token id to local index
    global2local_sid = {global_id: local_idx for local_idx, global_id in enumerate(sid_token_ids)}
    
    # Also load the item info mapping (SID -> item title/id) for HR/NDCG evaluation
    item_dict = {}
    sid_to_title = {}
    with open(args.info_file, 'r') as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip() for _ in items] # This is the SID sequence like <a_60><b_159><c_203>
        for i, name in enumerate(item_names):
            if name not in item_dict:
                item_dict[name] = [i]
            else:
                item_dict[name].append(i)

    # Build constraint hash dict (tree structure) just like Teacher
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
        
    # Identify the common prefix key for Step 1
    # We assume all items share the same prefix structure "### Response:\n"
    first_ID = prefixID[0]
    prefix_key = get_hash(first_ID[:prefix_index])
    valid_start_globals = hash_dict.get(prefix_key, [])
    valid_start_locals = [global2local_sid[g] for g in valid_start_globals if g in global2local_sid]
    print(f"Constraining Step 1 to {len(valid_start_locals)} valid start tokens (Key: {prefix_key})")

    # 2. Load Student Model
    print(f"Loading Student Model from {args.student_ckpt}...")
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    
    # Load weights
    checkpoint = torch.load(args.student_ckpt, map_location="cpu")
    student.load_state_dict(checkpoint['model_state_dict'])
    student.to(torch.bfloat16)
    student.to(device)
    student.eval()

    # 3. Dataset & DataLoader
    dataset = EvalSidDataset(
        train_file=args.test_file, 
        tokenizer=tokenizer, 
        category=args.category,
        max_len=1024,
        test=True # Important: test mode only returns input_ids and attention_mask
    )
    
    # Get all ground truths for evaluation
    test_data = dataset.get_all()

    def custom_collate(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
        
        # Left padding for inference
        max_len = max([len(seq) for seq in input_ids])
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for i in range(len(input_ids)):
            pad_len = max_len - len(input_ids[i])
            
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), input_ids[i]])
            )
            padded_attention_mask.append(
                torch.cat([torch.full((pad_len,), 0, dtype=torch.long), attention_mask[i]])
            )
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask)
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    # 4. Evaluation Loop
    print("Starting full 3-step evaluation by Student Model...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            batch_size = input_ids.shape[0]
            
            # Initialize beam candidates
            # Each candidate is a dict: {"tokens": [], "score": 0.0}
            # We maintain a list of candidates for each item in the batch
            # At step 0, we just have 1 empty candidate with score 0
            batch_candidates = [[{"tokens": [], "score": 0.0}] for _ in range(batch_size)]
            
            # Step 1: Predict <a_X>
            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs["logits_sid"][:, -1, :] # [batch_size, vocab_size]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            new_batch_candidates = []
            for b in range(batch_size):
                # For step 1, get allowed tokens from hash_dict using the prompt's hash
                # Wait, at step 1, hash_key is the prompt. But hash_dict uses prefixID.
                # Actually, step 1 is unconstrained among valid first tokens.
                # Let's get valid first tokens from hash_dict. The root hash is hash_number for prefixID[:prefix_index]
                # In the teacher's script, for step 1, sent[-prefix_index:] is used.
                # Since we don't have the exact prompt text generated here, we can just constrain 
                # based on all possible first tokens if needed, but it's easier to just pick top K and filter.
                
                # To be perfectly aligned with Teacher's ConstrainedLogitsProcessor:
                # The prompt ends with "### Response:\n". In tokenizer, this might be multiple tokens.
                # Let's just find the valid <a_X> tokens. Any token that starts a valid sequence is fine.
                
                # We'll just take the top 50 valid local tokens
                b_log_probs = log_probs[b]
                
                # Apply constraints for Step 1
                mask = torch.full_like(b_log_probs, float('-inf'))
                if valid_start_locals:
                    mask[valid_start_locals] = 0
                b_log_probs = b_log_probs + mask
                
                topk_vals, topk_indices = torch.topk(b_log_probs, args.top_k, dim=-1)
                
                candidates = []
                for k in range(args.top_k):
                    if topk_vals[k] == float('-inf'): continue # Skip invalid paths
                    local_idx = topk_indices[k].item()
                    global_id = local2global_sid[local_idx]
                    candidates.append({"tokens": [global_id], "score": topk_vals[k].item()})
                new_batch_candidates.append(candidates)
            
            batch_candidates = new_batch_candidates
            
            # Step 2: Predict <b_Y>
            # For each candidate in each batch, we append it to input_ids and forward
            # To batch this efficiently, we flatten the batch and candidates
            flat_input_ids = []
            flat_attention_mask = []
            flat_candidate_scores = []
            
            for b in range(batch_size):
                for cand in batch_candidates[b]:
                    flat_input_ids.append(torch.cat([input_ids[b], torch.tensor(cand["tokens"], device=device)]))
                    flat_attention_mask.append(torch.cat([attention_mask[b], torch.tensor([1], device=device)]))
                    flat_candidate_scores.append(cand["score"])
                    
            flat_input_ids = torch.stack(flat_input_ids)
            flat_attention_mask = torch.stack(flat_attention_mask)
            flat_candidate_scores = torch.tensor(flat_candidate_scores, device=device)
            
            outputs_2 = student(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
            next_token_logits_2 = outputs_2["logits_sid"][:, -1, :]
            log_probs_2 = F.log_softmax(next_token_logits_2, dim=-1)
            
            # Combine scores
            total_scores_2 = flat_candidate_scores.unsqueeze(1) + log_probs_2 # [batch_size * top_k, vocab_size]
            
            # Now we select top_k again for each original batch item
            new_batch_candidates = []
            for b in range(batch_size):
                start_idx = b * args.top_k
                end_idx = start_idx + args.top_k
                b_scores = total_scores_2[start_idx:end_idx, :] # [top_k, vocab_size]
                
                # Apply tree constraint
                # For each of the top_k candidates (which is 1 token long), find valid next tokens
                for c_idx in range(args.top_k):
                    cand_tokens = batch_candidates[b][c_idx]["tokens"]
                    # Teacher uses get_hash(sent[-count:]) for step > 0
                    hash_key = get_hash(cand_tokens)
                    allowed_globals = hash_dict.get(hash_key, [])
                    # Convert allowed globals to local indices
                    allowed_locals = [global2local_sid[g] for g in allowed_globals if g in global2local_sid]
                    
                    # Mask out disallowed tokens
                    mask = torch.full_like(b_scores[c_idx], float('-inf'))
                    if allowed_locals:
                        mask[allowed_locals] = 0
                    b_scores[c_idx] += mask
                
                # Flatten and get top K
                flat_b_scores = b_scores.view(-1)
                topk_vals, topk_flat_indices = torch.topk(flat_b_scores, args.top_k, dim=-1)
                
                candidates = []
                for k in range(args.top_k):
                    val = topk_vals[k].item()
                    if val == float('-inf'): continue # Dead end
                    flat_idx = topk_flat_indices[k].item()
                    cand_idx = flat_idx // len(sid_token_ids)
                    local_token_idx = flat_idx % len(sid_token_ids)
                    global_id = local2global_sid[local_token_idx]
                    
                    new_tokens = batch_candidates[b][cand_idx]["tokens"] + [global_id]
                    candidates.append({"tokens": new_tokens, "score": val})
                    
                # If less than top_k, pad (shouldn't happen with proper tree)
                while len(candidates) < args.top_k:
                    candidates.append({"tokens": [], "score": float('-inf')})
                    
                new_batch_candidates.append(candidates)
                
            batch_candidates = new_batch_candidates
            
            # Step 3: Predict <c_Z>
            flat_input_ids = []
            flat_attention_mask = []
            flat_candidate_scores = []
            valid_flat_indices = [] # Keep track of valid candidates
            
            for b in range(batch_size):
                for i, cand in enumerate(batch_candidates[b]):
                    if len(cand["tokens"]) == 2:
                        flat_input_ids.append(torch.cat([input_ids[b], torch.tensor(cand["tokens"], device=device)]))
                        flat_attention_mask.append(torch.cat([attention_mask[b], torch.tensor([1, 1], device=device)]))
                        flat_candidate_scores.append(cand["score"])
                        valid_flat_indices.append((b, i))
                        
            if len(flat_input_ids) > 0:
                flat_input_ids = torch.stack(flat_input_ids)
                flat_attention_mask = torch.stack(flat_attention_mask)
                flat_candidate_scores = torch.tensor(flat_candidate_scores, device=device)
                
                outputs_3 = student(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
                next_token_logits_3 = outputs_3["logits_sid"][:, -1, :]
                log_probs_3 = F.log_softmax(next_token_logits_3, dim=-1)
                
                total_scores_3 = flat_candidate_scores.unsqueeze(1) + log_probs_3
                
                # Re-distribute to batch
                b_scores_list = [ [] for _ in range(batch_size) ]
                for i, (b, c_idx) in enumerate(valid_flat_indices):
                    scores = total_scores_3[i]
                    cand_tokens = batch_candidates[b][c_idx]["tokens"]
                    hash_key = get_hash(cand_tokens)
                    allowed_globals = hash_dict.get(hash_key, [])
                    allowed_locals = [global2local_sid[g] for g in allowed_globals if g in global2local_sid]
                    
                    mask = torch.full_like(scores, float('-inf'))
                    if allowed_locals:
                        mask[allowed_locals] = 0
                    scores += mask
                    b_scores_list[b].append(scores)
                    
            # Finalize Top K for each batch
            for b in range(batch_size):
                if len(b_scores_list[b]) == 0:
                    all_predictions.append([])
                    continue
                    
                b_scores = torch.stack(b_scores_list[b]) # [num_valid_cands, vocab_size]
                flat_b_scores = b_scores.view(-1)
                topk_vals, topk_flat_indices = torch.topk(flat_b_scores, min(args.top_k, flat_b_scores.shape[0]), dim=-1)
                
                sample_preds = []
                for k in range(len(topk_vals)):
                    val = topk_vals[k].item()
                    if val == float('-inf'): continue
                    
                    flat_idx = topk_flat_indices[k].item()
                    cand_idx = flat_idx // len(sid_token_ids)
                    local_token_idx = flat_idx % len(sid_token_ids)
                    global_id = local2global_sid[local_token_idx]
                    
                    final_tokens = batch_candidates[b][cand_idx]["tokens"] + [global_id]
                    # Decode to string
                    pred_str = "".join([tokenizer.decode([t]) for t in final_tokens])
                    sample_preds.append(pred_str)
                    
                all_predictions.append(sample_preds)

    # 5. Evaluate Metrics (Full 3-token sequence)
    print("\n--- Full 3-Token Sequence Evaluation ---")
    topk_list = [1, 3, 5, 10, 20, 50]
    valid_topk = [k for k in topk_list if k <= args.top_k]
    ALLNDCG = [0.0] * len(valid_topk)
    ALLHR = [0.0] * len(valid_topk)
    
    for index, sample_preds in enumerate(all_predictions):
        target_item = test_data[index]['output'].strip(" \n\"")
        
        minID = 1000000
        for i, pred_token_str in enumerate(sample_preds):
            # The pred_token_str should now be a full 3-token sequence like "<a_X><b_Y><c_Z>"
            if pred_token_str == target_item:
                minID = i
                break
                
        for i, topk in enumerate(valid_topk):
            if minID < topk:
                ALLNDCG[i] += (1 / math.log(minID + 2))
                ALLHR[i] += 1
                
    num_samples = len(all_predictions)
    print(f"Evaluated on {num_samples} samples.")
    print(f"TopK: {valid_topk}")
    
    ndcg_res = [val / num_samples / (1.0 / math.log(2)) for val in ALLNDCG]
    hr_res = [val / num_samples for val in ALLHR]
    
    print(f"Full-Sequence NDCG: {[round(x, 4) for x in ndcg_res]}")
    print(f"Full-Sequence HR:   {[round(x, 4) for x in hr_res]}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({
            "metrics": {
                "topk": valid_topk,
                "full_sequence_ndcg": ndcg_res,
                "full_sequence_hr": hr_res
            },
            "sample_predictions": all_predictions  # Save all predictions for aggregation
        }, f, indent=4)
        
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()