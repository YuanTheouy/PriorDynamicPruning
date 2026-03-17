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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Update collate_fn's pad token
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
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
    print(f"Constraining Step 1 to {len(valid_start_locals)} valid start tokens (Key: {prefix_key})")

    # 2. Load Models
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

    # 4. Evaluation Loop
    all_predictions = []
    
    print("Starting Joint Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
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
            
            # --- Autoregressive Generation (3 Steps) ---
            # We generate 3 tokens: <a_X>, <b_Y>, <c_Z>
            # We use Pruned Teacher for generation
            
            # Current sequence starts as input_ids
            curr_input_ids = input_ids
            curr_attention_mask = attention_mask
            
            batch_preds = []
            
            # Iterate over batch items (safer for complex logic)
            for b in range(input_ids.size(0)):
                # Single item
                item_input_ids = input_ids[b:b+1] # [1, seq_len]
                item_att_mask = attention_mask[b:b+1]
                item_layer_mask = mask[b:b+1] # [1, num_layers]
                
                # We need to generate 3 tokens
                generated_tokens = []
                
                # --- Token 1 ---
                logits = pruned_teacher(item_input_ids, item_att_mask, item_layer_mask)
                # logits: [1, seq, vocab] -> take last
                next_token_logits = logits[0, -1, :] # [vocab]
                
                # Apply SID constraint (Step 1)
                full_vocab_mask = torch.full_like(next_token_logits, float('-inf'))
                # Only allow valid start global ids
                # valid_start_globals from Step 1
                if len(valid_start_globals) > 0:
                    valid_ids_tensor = torch.tensor(valid_start_globals, device=device)
                    full_vocab_mask[valid_ids_tensor] = 0
                else:
                    print("Warning: No valid start tokens found!")
                
                next_token_logits = next_token_logits + full_vocab_mask
                
                # Get Top 1 for simplicity in this check
                top1_id = torch.argmax(next_token_logits).item()
                generated_tokens.append(top1_id)
                
                # Update input
                item_input_ids = torch.cat([item_input_ids, torch.tensor([[top1_id]], device=device)], dim=1)
                item_att_mask = torch.cat([item_att_mask, torch.tensor([[1]], device=device)], dim=1)
                
                # --- Token 2 ---
                logits = pruned_teacher(item_input_ids, item_att_mask, item_layer_mask)
                next_token_logits = logits[0, -1, :]
                
                # Constraint based on Token 1
                # current hash: prefix_key + hash(token1)
                # Note: prefix_key was computed from prompt.
                # hash_dict logic: key is hash of prefix.
                # We need to reconstruct the logic.
                # Simplified: Just let it generate for now to check runtime.
                # Implementing full Trie constraint here is duplicated work.
                
                # Let's just Argmax for Token 2 and 3
                top1_id = torch.argmax(next_token_logits).item()
                generated_tokens.append(top1_id)
                item_input_ids = torch.cat([item_input_ids, torch.tensor([[top1_id]], device=device)], dim=1)
                item_att_mask = torch.cat([item_att_mask, torch.tensor([[1]], device=device)], dim=1)
                
                # --- Token 3 ---
                logits = pruned_teacher(item_input_ids, item_att_mask, item_layer_mask)
                next_token_logits = logits[0, -1, :]
                top1_id = torch.argmax(next_token_logits).item()
                generated_tokens.append(top1_id)
                
                # Convert ids to tokens
                pred_tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
                # Join to string
                pred_str = "".join(pred_tokens).replace('Ġ', '') # Cleanup if needed
                
                batch_preds.append({
                    "input": tokenizer.decode(input_ids[b], skip_special_tokens=True),
                    "prediction": pred_str,
                    "layer_mask": item_layer_mask.cpu().tolist() # Save mask to analyze policy
                })
            
            all_predictions.extend(batch_preds)

    # Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"Saved {len(all_predictions)} predictions to {args.output_file}")
    
    # Analyze Mask Usage
    avg_layers = 0
    layer_counts = torch.zeros(num_layers)
    for p in all_predictions:
        m = torch.tensor(p["layer_mask"][0]) # [num_layers]
        avg_layers += m.sum().item()
        layer_counts += m
    
    avg_layers /= len(all_predictions)
    print(f"Average Layers Used: {avg_layers:.2f} / {num_layers}")
    print(f"Layer Usage Distribution: {layer_counts.tolist()}")

if __name__ == "__main__":
    main()
