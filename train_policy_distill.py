import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from accelerate import Accelerator

from models.one_layer_student import OneLayerStudentModel
from models.router import LayerRouter
from models.pruned_teacher import PrunedTeacherWrapper
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
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to the pretrained student checkpoint")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--info_file", type=str, required=True, help="Path to item info txt file")
    parser.add_argument("--category", type=str, default="Office_Products")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k_layers", type=int, default=12, help="Number of layers to keep in Teacher")
    parser.add_argument("--output_dir", type=str, default="./policy_ckpts")
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
    
    sid_token_ids = get_sid_token_ids_from_info(tokenizer, args.info_file)
    if accelerator.is_main_process:
        print(f"Extracted {len(sid_token_ids)} active SID tokens.")
    
    # 2. Dataset & DataLoader
    dataset = SidSFTDataset(
        train_file=args.train_file, 
        tokenizer=tokenizer, 
        category=args.category,
        max_len=1024
    )
    
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
    if accelerator.is_main_process:
        print("Loading Teacher model...")
    # Load raw teacher
    raw_teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    raw_teacher.eval()
    
    # Wrap teacher for pruning
    # Note: PrunedTeacherWrapper keeps raw_teacher internally
    pruned_teacher = PrunedTeacherWrapper(raw_teacher)
    pruned_teacher.to(device)
    
    # Identify number of layers
    if hasattr(raw_teacher, "model") and hasattr(raw_teacher.model, "layers"):
        num_layers = len(raw_teacher.model.layers)
    elif hasattr(raw_teacher, "transformer") and hasattr(raw_teacher.transformer, "h"):
        num_layers = len(raw_teacher.transformer.h)
    else:
        # Fallback or error
        num_layers = 24 # Assumption
        if accelerator.is_main_process:
            print(f"Warning: Could not detect number of layers, assuming {num_layers}")

    if accelerator.is_main_process:
        print(f"Teacher has {num_layers} layers. Target Top-K: {args.top_k_layers}")

    # 4. Load Student (Frozen)
    if accelerator.is_main_process:
        print("Loading Student model...")
    student = OneLayerStudentModel(args.teacher_model, sid_token_ids)
    
    # Load checkpoint
    checkpoint = torch.load(args.student_ckpt, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        student.load_state_dict(checkpoint['model_state_dict'])
    else:
        student.load_state_dict(checkpoint)
        
    student.to(torch.bfloat16)
    student.eval()
    student.to(device)
    for param in student.parameters():
        param.requires_grad = False

    # 5. Initialize Router (Policy Network)
    # Input size is hidden size of student
    hidden_size = student.backbone.config.hidden_size
    router = LayerRouter(hidden_size=hidden_size, num_layers=num_layers, top_k=args.top_k_layers)
    router.to(torch.bfloat16) # Match precision
    router.to(device)
    router.train()

    # Optimizer - Only optimize Router
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)

    # Prepare with Accelerate
    # We don't prepare teacher or student as they are frozen/eval
    router, optimizer, dataloader = accelerator.prepare(router, optimizer, dataloader)

    # 6. Training Loop
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
            
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Filter valid positions
            valid_positions_mask = (labels != -100)
            if not valid_positions_mask.any():
                continue

            # --- 1. Get Student State ---
            with torch.no_grad():
                student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = student_outputs["last_hidden_state"]
                
                # Use the state of the LAST token in the sequence (or prompt end)
                # Here we use the last token of the input sequence
                # shape: [batch, hidden_size]
                # Note: input_ids includes prompt + target?
                # In SFTDataset, input_ids is full sequence.
                # However, for routing, we want to route based on the context.
                # Since we use Sequence-level mask, we can use the representation of the last token.
                # Or pooling. Let's stick to last token as discussed.
                
                # To get the true last token (ignoring padding), we use attention_mask
                # last_indices = attention_mask.sum(dim=1) - 1
                # state = last_hidden_state[torch.arange(batch_size), last_indices]
                
                # Simpler approximation if left-padded or right-padded correctly:
                # If right-padded (standard), last non-pad token.
                # If we assume `input_ids` are right-padded (collator does that).
                last_indices = attention_mask.sum(dim=1) - 1
                state = last_hidden_state[torch.arange(input_ids.size(0), device=device), last_indices]

            # --- 2. Router Forward ---
            # mask: [batch, num_layers]
            mask, scores = router(state)

            # --- 3. Full Teacher Forward (Target) ---
            with torch.no_grad():
                # We can call raw_teacher directly
                full_outputs = raw_teacher(input_ids=input_ids, attention_mask=attention_mask)
                full_logits = full_outputs.logits
                
                # Shift and filter for SID
                shift_full_logits = full_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_valid_mask = (shift_labels != -100)
                
                active_full_logits = shift_full_logits[shift_valid_mask]
                active_full_sid_logits = active_full_logits[:, sid_token_ids]
                
                # Target Probs
                target_probs = F.softmax(active_full_sid_logits / args.temperature, dim=-1)

            # --- 4. Pruned Teacher Forward (Prediction) ---
            # We use the wrapper. It requires layer_mask.
            pruned_logits = pruned_teacher(input_ids, attention_mask, mask)
            
            # Shift and filter
            shift_pruned_logits = pruned_logits[..., :-1, :].contiguous()
            # Re-use mask and indices
            active_pruned_logits = shift_pruned_logits[shift_valid_mask]
            active_pruned_sid_logits = active_pruned_logits[:, sid_token_ids]
            
            # Log Probs for KL
            log_pruned_probs = F.log_softmax(active_pruned_sid_logits / args.temperature, dim=-1)

            # --- 5. Loss ---
            loss = F.kl_div(log_pruned_probs, target_probs, reduction='batchmean') * (args.temperature ** 2)
            
            # Backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
            
            # Save Checkpoint
            save_path = os.path.join(args.output_dir, f"policy_epoch_{epoch+1}.pt")
            unwrapped_router = accelerator.unwrap_model(router)
            torch.save(unwrapped_router.state_dict(), save_path)
            print(f"Saved policy checkpoint to {save_path}")

if __name__ == "__main__":
    main()
