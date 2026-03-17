import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
import argparse
from tqdm import tqdm
import json
import math
import numpy as np

from models.one_layer_student import OneLayerStudentModel
from utils_distill import get_sid_token_ids_from_info
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint")
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to the trained student checkpoint (.pt file)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="./results/joint_result.json")
    parser.add_argument("--top_k_student", type=int, default=50, help="Number of branches to keep from student")
    parser.add_argument("--num_beams", type=int, default=50, help="Number of beams for teacher generation")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if args.teacher_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    # Load info file and build hash dict
    with open(args.info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]

    if args.teacher_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]

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

    # Extract active SID tokens for student
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    local2global_sid = {local_idx: global_id for local_idx, global_id in enumerate(sid_token_ids)}

    # 2. Load Models
    print("Loading Teacher Model...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    teacher.eval()
    teacher.to(device)

    print(f"Loading Student Model from {args.student_ckpt}...")
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    checkpoint = torch.load(args.student_ckpt, map_location="cpu")
    student.load_state_dict(checkpoint['model_state_dict'])
    student.to(torch.bfloat16)
    student.to(device)
    student.eval()

    # 3. Dataset
    dataset = EvalSidDataset(
        train_file=args.test_file, 
        tokenizer=tokenizer, 
        category=args.category,
        max_len=2560,
        test=True
    )
    test_data = dataset.get_all()

    def custom_collate(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
        
        max_len = max([len(seq) for seq in input_ids])
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for i in range(len(input_ids)):
            pad_len = max_len - len(input_ids[i])
            padded_input_ids.append(torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), input_ids[i]]))
            padded_attention_mask.append(torch.cat([torch.full((pad_len,), 0, dtype=torch.long), attention_mask[i]]))
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask)
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    # 4. Joint Evaluation
    print("Starting Joint Inference (Student -> Teacher)...")
    outputs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.shape[0]
            maxLen = input_ids.shape[1]

            # --- STEP 1: Student predicts Top-K first tokens ---
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = student_outputs["logits_sid"][:, -1, :] # [batch_size, len(sid_token_ids)]
            
            # Get top K global token IDs for each sample in the batch
            topk_probs, topk_indices = torch.topk(F.softmax(next_token_logits, dim=-1), args.top_k_student, dim=-1)
            
            # Convert local indices to global token IDs
            batch_student_allowed_tokens = []
            for b in range(batch_size):
                allowed_global_ids = [local2global_sid[topk_indices[b, k].item()] for k in range(args.top_k_student)]
                batch_student_allowed_tokens.append(set(allowed_global_ids))

            # --- STEP 2: Teacher generates using Student's constraints ---
            # Define dynamic prefix allowed function
            def joint_prefix_allowed_tokens_fn(batch_id, hash_key):
                # hash_key is passed as list of ints
                hash_number = get_hash(hash_key)
                teacher_allowed = hash_dict.get(hash_number, [])
                
                # Check if it's the first step by the length of hash_key
                # At step 0, hash_key is sent[-prefix_index:], so length is prefix_index
                if len(hash_key) == prefix_index:
                    # Intersect Teacher's valid next tokens with Student's top-K
                    student_allowed = batch_student_allowed_tokens[batch_id]
                    final_allowed = [t for t in teacher_allowed if t in student_allowed]
                    # Fallback in case intersection is empty (should rarely happen if student is good)
                    if not final_allowed:
                        return teacher_allowed
                    return final_allowed
                else:
                    # For subsequent steps, just use Teacher's original constraints
                    return teacher_allowed

            generation_config = GenerationConfig(
                num_beams=args.num_beams,
                length_penalty=1.0,
                num_return_sequences=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                top_k=None,
                top_p=None,
            )

            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=joint_prefix_allowed_tokens_fn,
                num_beams=args.num_beams,
                base_model=args.teacher_model,
                eos_token_id=tokenizer.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = teacher.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

            batched_completions = generation_output.sequences[:, maxLen:]
            
            if args.teacher_model.lower().find("llama") > -1:
                batch_output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                batch_output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
                
            batch_output = [_.split("Response:\n")[-1].strip() for _ in batch_output]
            real_outputs = [batch_output[i * args.num_beams: (i + 1) * args.num_beams] for i in range(batch_size)]
            
            outputs_list.extend(real_outputs)

    # 5. Save Results
    for i, test in enumerate(test_data):
        test["predict"] = outputs_list[i]
        if 'dedup' in test:
            test.pop('dedup')

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(test_data, f, indent=4)
        
    print(f"Results saved to {args.output_file}")
    print("Now you can run 'python calc.py' on the output file to get the final metrics!")

if __name__ == "__main__":
    main()