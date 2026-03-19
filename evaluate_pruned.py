
import sys
import os

# [CRITICAL] Inject local transformers library dynamically based on script location
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_src_path = os.path.join(current_dir, "transformers", "src")
sys.path.insert(0, transformers_src_path)

import transformers
print(f"DEBUG: Transformers library path: {transformers.__file__}")

import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
import random
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    top_k_layers: int = 28,  # [NEW] Control how many layers to keep
    strategy: str = "uniform" # [NEW] Pruning strategy
):
    random.seed(seed)
    set_seed(seed)
    # Note: run_test.sh sets CUDA_VISIBLE_DEVICES externally, we don't hardcode it here
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    category = category_dict[category]
    print(f"Testing Category: {category} with Strategy: {strategy} keeping {top_k_layers} layers")

    # Load Model (Will use local transformers AutoModelForCausalLM which resolves to Qwen2ForCausalLM)
    # Using attn_implementation="sdpa" to ensure fast evaluation
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    
    # Calculate Layer Mask
    num_layers = len(model.model.layers)
    mask = torch.zeros(num_layers)
    if strategy == "uniform":
        indices = np.linspace(0, num_layers - 1, top_k_layers, dtype=int)
        mask[indices] = 1.0
    elif strategy == "first_k":
        mask[:top_k_layers] = 1.0
    elif strategy == "last_k":
        mask[-top_k_layers:] = 1.0
    else:
        # Default fallback
        mask[:] = 1.0
        
    print(f"Layer Mask: {mask.tolist()}")
    # Move mask to device
    layer_mask = mask.to(dtype=torch.bfloat16).to(device).unsqueeze(0) # [1, num_layers]

    # [CRITICAL FIX] Mount to model.config instead of model.model to survive device_map and accelerate
    model.config.custom_layer_mask = mask.tolist()

    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
        
    if base_model.lower().find("gpt2") > -1:
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
        hash_number = get_hash(ID[prefix_index:])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            batch_size = len(padding_encodings["input_ids"])
            
            # We ALREADY mounted layer_mask to model.model.layer_mask in main()
            # DO NOT pass it to generate(), otherwise _validate_model_kwargs will crash!
            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False, # Save memory
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
    
    for idx, encodings in enumerate(tqdm(new_encodings)):
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        outputs = outputs + output
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
  
    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)
