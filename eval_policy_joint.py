import sys
import os

# [CRITICAL] Inject local transformers library dynamically based on script location
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

import transformers
print(f"DEBUG: Transformers library path: {transformers.__file__}")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
import argparse
from tqdm import tqdm
import json
import math
from accelerate import Accelerator

from models.one_layer_student import OneLayerStudentModel
from models.router import LayerRouter
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor

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
    
    from transformers import Qwen2ForCausalLM
    # Important: use device_map="auto" or explicitly map to accelerator device
    # so that the model parameters are actually moved to the correct GPU
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.to(device)
    raw_teacher.eval()
    
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        num_layers = 24
    if accelerator.is_main_process:
        print(f"Teacher has {num_layers} layers.")

    raw_teacher.config.pad_token_id = tokenizer.eos_token_id
    raw_teacher.config.eos_token_id = tokenizer.eos_token_id
    raw_teacher.config.bos_token_id = tokenizer.bos_token_id
    
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

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)
            
            # --- Step 1: Get Router Mask via Student ---
            student_out = student(input_ids, attention_mask)
            last_hidden_state = student_out["last_hidden_state"]
            last_indices = attention_mask.sum(dim=1) - 1
            state = last_hidden_state[torch.arange(batch_size, device=device), last_indices]
            mask, _ = router(state) # [batch, num_layers] Tensor
            
            # Convert mask to 2D list for modeling_qwen2.py
            list_mask = mask.cpu().tolist()
            raw_teacher.config.custom_layer_mask = list_mask

            # --- Step 2: Native model.generate() with Beam Search ---
            generation_config = GenerationConfig(
                num_beams=args.top_k_items,
                length_penalty=1.0,
                num_return_sequences=args.top_k_items,
                pad_token_id = raw_teacher.config.pad_token_id,
                eos_token_id = raw_teacher.config.eos_token_id,
                max_new_tokens = 256,
                top_k=None,
                top_p=None,
            )
            
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_semantic,
                num_beams=args.top_k_items,
                base_model=args.teacher_model,
                eos_token_id=raw_teacher.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])
            
            generation_output = raw_teacher.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=logits_processor,
            )
            
            # Clear mask
            raw_teacher.config.custom_layer_mask = None
            
            # --- Step 3: Decode Outputs ---
            batched_completions = generation_output.sequences[:, max_len:]
            
            if args.teacher_model.lower().find("llama") > -1:
                output_strs = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                output_strs = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
                
            output_strs = [_.split("Response:\n")[-1].strip() for _ in output_strs]
            
            # Group by batch_size (each input has num_beams outputs)
            for b in range(batch_size):
                start_idx = b * args.top_k_items
                end_idx = (b + 1) * args.top_k_items
                sample_preds = output_strs[start_idx:end_idx]
                
                all_predictions.append({
                    "input": tokenizer.decode(input_ids[b], skip_special_tokens=True),
                    "sample_predictions": sample_preds,
                    "layer_mask": list_mask[b]
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
