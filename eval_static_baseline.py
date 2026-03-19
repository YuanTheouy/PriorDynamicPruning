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
    # evaluate.py does NOT do this immediately, but does it later. 
    # Let's match evaluate.py exactly.
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
    # FIX: evaluate.py uses device_map="auto" and loads model via AutoModelForCausalLM
    # We must ensure Qwen2ForCausalLM loads exactly the same way.
    raw_teacher = Qwen2ForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.to(device)
    raw_teacher.eval()
    
    # Make sure pad_token_id is correctly set in model config
    raw_teacher.config.pad_token_id = tokenizer.pad_token_id
    raw_teacher.config.eos_token_id = tokenizer.eos_token_id
    # evaluate.py does: model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    # AND model.config.bos_token_id = tokenizer.bos_token_id
    raw_teacher.config.bos_token_id = tokenizer.bos_token_id
    
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

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=256,  # Fix: Match evaluate.py
            length_penalty=0.0,  # Fix: Match evaluate.py
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 

        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = raw_teacher.config.pad_token_id,
            eos_token_id = raw_teacher.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_semantic,
                num_beams=num_beams,
                base_model=args.teacher_model,
                eos_token_id=raw_teacher.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            batch_size = len(padding_encodings["input_ids"])
            # Use fixed mask for all samples in batch
            mask = static_mask.repeat(batch_size, 1) # [batch, num_layers]

            # ================= DEBUG INFO START ================
            if not hasattr(evaluate, "debug_printed") and accelerator.is_main_process:
                print("\n================ EVAL_STATIC_BASELINE DEBUG INFO START ================")
                print(f"[Model Config] bos_token_id: {raw_teacher.config.bos_token_id}, eos_token_id: {raw_teacher.config.eos_token_id}, pad_token_id: {raw_teacher.config.pad_token_id}")
                print(f"[Tokenizer] bos: {tokenizer.bos_token_id}, eos: {tokenizer.eos_token_id}, pad: {tokenizer.pad_token_id}")
                
                input_tensor = torch.tensor(padding_encodings["input_ids"])
                print(f"[Input IDs] Shape: {input_tensor.shape}")
                print(f"[Input IDs Sample 0 (Last 15 tokens)]: {input_tensor[0, -15:].tolist()}")
                
                attn_tensor = torch.tensor(attention_mask)
                print(f"[Attention Mask Sample 0 (Last 15 tokens)]: {attn_tensor[0, -15:].tolist()}")
                
                print(f"[Hash Dict Size] Semantic: {len(hash_dict)}")
                print(f"[Generation Config] num_beams: {generation_config.num_beams}, max_new_tokens: {generation_config.max_new_tokens}, length_penalty: {generation_config.length_penalty}")
                print("================ EVAL_STATIC_BASELINE DEBUG INFO END ================\n")
                evaluate.debug_printed = True
            # ================= DEBUG INFO END ================

            generation_output = raw_teacher.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=logits_processor,
                layer_mask=mask,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if args.teacher_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs, padding_encodings["input_ids"]

    # 5. Inference Loop
    all_predictions = []
    
    if accelerator.is_main_process:
        print("Starting Static Baseline Inference (Accelerated via model.generate)...")
    
    # We need to manually batch to use evaluate function
    # Instead of dataloader, we iterate over dataset manually to match evaluate.py
    encodings = [dataset[i] for i in range(len(dataset))]
    new_encodings = []
    BLOCK = (len(encodings) + args.batch_size - 1) // args.batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * args.batch_size: (i + 1) * args.batch_size])

    if accelerator.is_main_process:
        pbar = tqdm(new_encodings)
    else:
        pbar = new_encodings

    for batch_encodings in pbar:
        # Use standard evaluation
        real_outputs, padded_input_ids = evaluate(
            batch_encodings, 
            max_new_tokens=256, # Fix: Match evaluate.py
            num_beams=args.top_k_items, 
            length_penalty=0.0  # Fix: Match evaluate.py
        )
        
        for i, preds in enumerate(real_outputs):
            input_ids_clean = [t for t in padded_input_ids[i] if t != tokenizer.pad_token_id]
            input_str = tokenizer.decode(input_ids_clean, skip_special_tokens=True)
            all_predictions.append({
                "input": input_str,
                "sample_predictions": preds,
                "layer_mask": static_mask_cpu.tolist()
            })

    rank_output_file = args.output_file.replace(".json", f"_rank{accelerator.process_index}.json")
    os.makedirs(os.path.dirname(rank_output_file), exist_ok=True)
    with open(rank_output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"[Rank {accelerator.process_index}] Saved {len(all_predictions)} predictions to {rank_output_file}")
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
