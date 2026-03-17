import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm

from models.one_layer_student import OneLayerStudentModel
from utils_distill import get_sid_token_ids_from_info
from data import SidSFTDataset

def collate_fn(batch):
    # Filter out None values that might come from dataset
    batch = [b for b in batch if b is not None]
    
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0) # Update pad value based on tokenizer later
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model checkpoint")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./student_ckpts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Tokenizer & SID Vocab
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    
    # Create a mapping from global token id to local sid index
    global2local_sid = {global_id: local_idx for local_idx, global_id in enumerate(sid_token_ids)}

    # 2. Dataset & DataLoader
    dataset = SidSFTDataset(
        train_file=args.train_file, 
        tokenizer=tokenizer, 
        category=args.category,
        max_len=1024
    )
    
    # Custom collate_fn to handle padding
    def custom_collate(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)

    # 3. Load Teacher
    print("Loading Teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    teacher.eval()
    teacher.to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    # 4. Load & Initialize Student
    print("Initializing Student model...")
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    student.init_from_teacher(teacher)
    student.to(device)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # 5. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mask indicating which positions are targets (i.e. predicting SID tokens)
            # labels is -100 for prompt, and contains the actual token id for the target
            # Note: The model predicts the *next* token, so we need to shift labels or use causal logic.
            # In standard causal LM, logits at pos t predict token at pos t+1.
            # So if label is valid at pos t, we compute loss against logits at pos t-1.
            
            # valid_positions shape: [batch, seq_len]
            # It's True where label != -100.
            valid_positions_mask = (labels != -100)
            
            if not valid_positions_mask.any():
                continue

            # --- Teacher Forward ---
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                # logits: [batch, seq_len, vocab_size]
                teacher_logits = teacher_outputs.logits
                
                # Shift logits and labels for causal LM
                # Logits at i predict label at i (after shifting labels)
                shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_valid_mask = (shift_labels != -100)
                
                # Filter to only keep active positions
                # active_teacher_logits: [num_active_tokens, vocab_size]
                active_teacher_logits = shift_teacher_logits[shift_valid_mask]
                
                # Keep only SID vocab
                # active_teacher_sid_logits: [num_active_tokens, len(sid_token_ids)]
                active_teacher_sid_logits = active_teacher_logits[:, sid_token_ids]
                
                # Soft probabilities
                teacher_probs = F.softmax(active_teacher_sid_logits / args.temperature, dim=-1)

            # --- Student Forward ---
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            # student_logits: [batch, seq_len, len(sid_token_ids)]
            student_logits_sid = student_outputs["logits_sid"]
            
            shift_student_logits = student_logits_sid[..., :-1, :].contiguous()
            active_student_logits = shift_student_logits[shift_valid_mask]
            
            # Log probabilities for KL Div
            log_student_probs = F.log_softmax(active_student_logits / args.temperature, dim=-1)
            
            # --- Calculate KL Divergence Loss ---
            # KLDivLoss expects input as log-probs and target as probs
            loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean') * (args.temperature ** 2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"student_epoch_{epoch+1}.pt")
        torch.save({
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sid_token_ids': sid_token_ids
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
