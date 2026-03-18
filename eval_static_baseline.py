import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
import argparse
from tqdm import tqdm
import json
import numpy as np
from accelerate import Accelerator

from LogitProcessor import ConstrainedLogitsProcessor
from models.pruned_teacher import PrunedTeacherWrapper
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    
    # Extract input_ids from the dataset output
    # Note: evaluate.py uses 'input_ids' from EvalSidDataset directly
    input_ids = [b["input_ids"] for b in batch]
    
    # Let padding be handled correctly by finding max len in batch
    max_len = max([len(ids) for ids in input_ids])
    
    # Important: In evaluate.py, tokenizer.pad_token_id is used.
    # We rely on the global pad_token_id set in main()
    global pad_token_id 
    
    padded_input_ids = []
    attention_mask = []
    
    for ids in input_ids:
        # Convert to list if it's a tensor
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        L = len(ids)
        # Left padding as required by HF generate for decoder-only models
        # Match evaluate.py logic: [pad] * (max - L) + ids
        # Use simple list concatenation then tensor conversion for consistency
        padded_ids = [pad_token_id] * (max_len - L) + ids
        padded_mask = [0] * (max_len - L) + [1] * L
        
        padded_input_ids.append(torch.tensor(padded_ids, dtype=torch.long))
        attention_mask.append(torch.tensor(padded_mask, dtype=torch.long))
        
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_mask)
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
    
    # CRITICAL FIX: evaluate.py seems to be using 151645 for padding (repetition_penalty/eos related?)
    # or tokenizer.pad_token_id might be different.
    # From debug log: evaluate.py uses 151645.
    # Let's force check what tokenizer.pad_token_id is, or align it.
    # Qwen2 tokenizer: <|endoftext|> is 151643. 151645 is <|im_start|>? No.
    # Let's trust the tokenizer unless we see divergence.
    # Wait, in the user provided log for eval_static_baseline.py:
    # DEBUG: Input IDs Sample 0: [151645, 151645, ...]
    # So eval_static_baseline.py IS ALREADY using 151645!
    
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # 2. Dataset
    dataset = EvalSidDataset(args.test_file, tokenizer, max_len=1024, test=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # 3. Load Teacher
    if accelerator.is_main_process:
        print("Loading Teacher model...")
    # from transformers import Qwen2ForCausalLM
    from models.modeling_qwen2 import Qwen2ForCausalLM
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    # Explicitly move model to device (Accelerator usually handles this via prepare, but we are using raw_teacher for generate)
    # Since we are not using accelerator.prepare_model(raw_teacher), we must move it manually.
    raw_teacher.to(device)
    raw_teacher.eval()
    
    # Make sure pad_token_id is correctly set in model config
    raw_teacher.config.pad_token_id = tokenizer.pad_token_id
    raw_teacher.config.eos_token_id = tokenizer.eos_token_id
    
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        num_layers = 24
    
    # PrunedTeacherWrapper is not needed if we use Qwen2ForCausalLM directly
    # pruned_teacher = PrunedTeacherWrapper(raw_teacher)
    # pruned_teacher.to(device)

    # 4. Construct Static Mask
    static_mask_cpu = get_static_mask(args.strategy, num_layers, args.top_k_layers)
    # Ensure static mask has correct dtype (BFloat16) to match model
    static_mask = static_mask_cpu.to(dtype=torch.bfloat16).to(device).unsqueeze(0) # [1, num_layers]
    
    if accelerator.is_main_process:
        print(f"Static Mask: {static_mask_cpu.tolist()}")
        print(f"Active Layers: {torch.nonzero(static_mask_cpu).squeeze().tolist()}")

    # Trie Logic - Build Hash Dict for ConstrainedLogitsProcessor
    item_dict = {}
    with open(args.info_file, 'r') as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip() for _ in items]
    info_semantic = [f'''### Response:\n{_}\n''' for _ in item_names] # Original format: ends with \n
    # evaluate.py uses: info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
    # But wait! evaluate.py semantic_ids ALREADY have "\n" appended:
    # semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
    # So effectively it is "### Response:\nSID\n"
    
    # Let's align exactly with evaluate.py
    info_semantic = [f'''### Response:\n{_}\n''' for _ in item_names] 
    
    if args.teacher_model.lower().find("llama") > -1:
        # evaluate.py: prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        # evaluate.py: prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        
    if args.teacher_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
        
    hash_dict = dict()
    # evaluate.py: 
    # for index, ID in enumerate(prefixID):
    #     ID.append(tokenizer.eos_token_id)
    #     for i in range(prefix_index, len(ID)): ...
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
        
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        # Convert input_ids to list if it is a tensor
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
            
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    # 5. Inference Loop with Native generate()
    all_predictions = []
    
    if accelerator.is_main_process:
        print("Starting Static Baseline Inference (Accelerated via model.generate)...")
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
        
    generation_config = GenerationConfig(
        num_beams=args.top_k_items,
        num_return_sequences=args.top_k_items,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=3,  # Since SID is 3 tokens long
        top_k=None,
        top_p=None,
    )

    debug_cnt = 0

    with torch.no_grad():
        for batch in pbar:
            # IMPORTANT FIX: In evaluate.py, num_beams=50 is passed to ConstrainedLogitsProcessor.
            # But the input_ids passed to it are NOT yet expanded by beam search in the first step?
            # Actually, `model.generate` handles beam search expansion internally.
            # However, LogitsProcessor is called AFTER beam expansion.
            # So `input_ids` seen by LogitsProcessor will have shape (batch_size * num_beams, seq_len).
            # Our `ConstrainedLogitsProcessor` expects this structure and reshapes it:
            # input_ids.view(-1, self._num_beams, input_ids.shape[-1])
            
            # So we must ensure `num_beams` passed to CLP matches `generation_config.num_beams`.
            # args.top_k_items is used for both. This seems correct.
            
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_semantic,
                num_beams=args.top_k_items,
                base_model=args.teacher_model,
                eos_token_id=tokenizer.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)
            
            # Use fixed mask for all samples in batch
            mask = static_mask.repeat(batch_size, 1) # [batch, num_layers]
            
            # CRITICAL: If using beam search, layer_mask needs to be expanded to match beam size!
            # model.generate expands input_ids and attention_mask, but it DOES NOT know how to expand `layer_mask` automatically
            # unless we hook into `prepare_inputs_for_generation` properly or pass it pre-expanded?
            # NO, `prepare_inputs_for_generation` is called inside generate.
            # If we pass `layer_mask` via kwargs to generate, it is passed to `prepare_inputs_for_generation`.
            # Let's check `models/modeling_qwen2.py`:
            # `prepare_inputs_for_generation` receives `layer_mask` and puts it into `model_inputs`.
            # Then `model.forward` receives `model_inputs`.
            # Inside `generate`, the inputs are expanded for beam search (interleave_repeat).
            # Does `generate` expand arbitrary kwargs? 
            # Usually NO. It expands `input_ids`, `attention_mask`, `token_type_ids`, etc.
            # BUT it expands `model_kwargs`.
            # If we pass `layer_mask` as a kwarg to `generate`, it goes into `model_kwargs`.
            # `GenerationMixin._expand_inputs_for_generation` handles expansion.
            # It expands items in `model_kwargs` if they match batch size.
            
            # Let's verify if `layer_mask` (batch_size, num_layers) is correctly expanded to (batch_size * num_beams, num_layers).
            # If not, the model will see a mismatch in batch dimension during beam search!
            # Batch size is 8. Num beams is 50.
            # Forward pass 1: input (8*50, seq_len). layer_mask (8, 28).
            # ERROR or Broadcasting?
            # In modeling_qwen2.py:
            # layer_mask_i = layer_mask[:, idx] -> (8,)
            # hidden_states -> (400, seq, hidden)
            # layer_mask_i * hidden_states -> (8, 1, 1) * (400, ...) -> Broadcasting?
            # (8, 1, 1) broadcasts to (8, ..., ...). 
            # But (400, ...) cannot broadcast with (8, ...)! 400 is not a multiple of 8 in a way that aligns unless 400 % 8 == 0.
            # Wait, 400 = 8 * 50.
            # PyTorch broadcasting: (8, 1, 1) and (400, S, H). 
            # 8 != 400. This should FAIL with shape mismatch!
            
            # Why did it NOT fail?
            # Maybe `generate` IS expanding it?
            # Or maybe `layer_mask` became None?
            # Or maybe `forward` wasn't called with the mask?
            
            # Let's proactively expand it to be safe.
            # The correct behavior is to repeat each element `num_beams` times (interleave).
            # [A, B] -> [A, A, ..., B, B, ...]
            expanded_mask = mask.repeat_interleave(args.top_k_items, dim=0) # [batch * beams, num_layers]
            
            # DEBUG PRINT
            if accelerator.is_main_process and debug_cnt == 0:
                print(f"DEBUG: Input IDs Shape: {input_ids.shape}")
                print(f"DEBUG: Input IDs Sample 0: {input_ids[0].tolist()}")
                print(f"DEBUG: Attention Mask Sample 0: {attention_mask[0].tolist()}")
                print(f"DEBUG: Layer Mask Shape (Original): {mask.shape}")
                print(f"DEBUG: Layer Mask Shape (Expanded): {expanded_mask.shape}")
                debug_cnt += 1

            # We call the underlying raw_teacher (Qwen2ForCausalLM), passing our custom layer_mask!
            generation_output = raw_teacher.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=logits_processor,
                layer_mask=expanded_mask, # Pass the expanded mask!
            )
            
            # Extract generated tokens
            batched_completions = generation_output.sequences[:, max_len:]
            
            # DEBUG PRINT GENERATION
            if accelerator.is_main_process and debug_cnt == 1:
                print(f"DEBUG: Generated Sequences Shape: {generation_output.sequences.shape}")
                print(f"DEBUG: Generated Sample 0: {generation_output.sequences[0, max_len:].tolist()}")
                debug_cnt += 1

            # Group back into batch_size x top_k_items
            batched_completions = batched_completions.view(batch_size, args.top_k_items, -1)
            
            for b in range(batch_size):
                sample_preds = []
                for k in range(args.top_k_items):
                    pred_tokens = batched_completions[b, k].tolist()
                    # Clean up padding/eos tokens
                    pred_tokens = [t for t in pred_tokens if t != tokenizer.eos_token_id and t != tokenizer.pad_token_id]
                    pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True).replace('Ġ', '')
                    sample_preds.append(pred_str)
                
                # Original input string
                input_ids_clean = [int(t) for t in input_ids[b].tolist() if t != tokenizer.pad_token_id]
                input_str = tokenizer.decode(input_ids_clean, skip_special_tokens=True)
                
                all_predictions.append({
                    "input": input_str,
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
