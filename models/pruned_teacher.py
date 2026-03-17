import torch
import torch.nn as nn
from transformers import PreTrainedModel

class PrunedTeacherWrapper(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask, layer_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            layer_mask: [batch_size, num_layers] - binary mask (0 or 1)
        """
        # 1. Embeddings
        # Access the base model (e.g. LlamaModel, Qwen2Model)
        # Usually teacher.model or teacher.transformer
        if hasattr(self.teacher, "model"):
            base_model = self.teacher.model
        elif hasattr(self.teacher, "transformer"):
            base_model = self.teacher.transformer
        else:
            raise ValueError("Unknown teacher model structure")

        # Get embeddings
        if hasattr(base_model, "embed_tokens"):
            hidden_states = base_model.embed_tokens(input_ids)
        elif hasattr(base_model, "wte"):
             hidden_states = base_model.wte(input_ids)
        else:
            raise ValueError("Unknown embedding layer")

        batch_size, seq_length = input_ids.shape
        
        # 2. Position Embeddings (RoPE) and Attention Mask
        
        # --- Handle Attention Mask ---
        # The error "The expanded size of the tensor (125) must match the existing size (8) at non-singleton dimension 2"
        # indicates a shape mismatch in broadcasting.
        # This usually happens when _prepare_decoder_attention_mask creates a 4D mask [batch, 1, seq_len, seq_len],
        # but the attention implementation (SDPA) expects a different shape or broadcasting behavior.
        
        # Qwen2's implementation might be using SDPA which handles broadcasting differently depending on PyTorch version
        # and whether a causal mask is automatically applied.
        
        # If we use _prepare_decoder_attention_mask, it returns a [batch, 1, seq_len, seq_len] mask.
        # However, for SDPA with causal masking, sometimes we just need the padding mask [batch, seq_len]?
        # Or a causal mask is needed.
        
        # Let's try to trust the model's internal preparation fully, BUT:
        # If the error is about broadcasting [8, 12, 125, 125] vs [8, 125], it implies the mask is [8, 1, 125, 125]
        # and something else is [8, 125].
        # Wait, "Target sizes: [8, 12, 125, 125]. Tensor sizes: [8, 125]"
        # This suggests the `attention_mask` being passed is likely the raw [8, 125] tensor, NOT the expanded one.
        
        # Wait, if `_prepare_decoder_attention_mask` was called successfully, `extended_attention_mask` should be 4D.
        # If it wasn't called (e.g. method doesn't exist on this version of Qwen/Transformers?), we fell back to boolean mask [8, 125].
        # SDPA might be trying to broadcast [8, 125] against [8, 12, 125, 125] (batch, heads, q, k).
        # [8, 125] broadcasts to [8, 1, 1, 125] -> [8, 12, 125, 125]? No.
        # [8, 125] usually means [batch, key_len].
        
        # Let's explicitly look for `_prepare_decoder_attention_mask`.
        # If it fails, we need to manually create the causal mask + padding mask.
        
        extended_attention_mask = None
        if hasattr(base_model, "_prepare_decoder_attention_mask"):
             # This returns [batch, 1, tgt_len, src_len]
             extended_attention_mask = base_model._prepare_decoder_attention_mask(
                 attention_mask, 
                 (batch_size, seq_length), 
                 hidden_states, 
                 0 # past_key_values_length
             )
        else:
            # Fallback: Create 4D causal mask manually
            # This is safer for SDPA in many HF models which expect the full mask if passed.
            # Mask shape: [batch, 1, seq_len, seq_len]
            
            # 1. Create Causal Mask (Lower Triangular)
            # [seq_len, seq_len]
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device, dtype=torch.bool))
            
            # 2. Combine with Padding Mask
            # attention_mask is [batch, seq_len] (1 for keep, 0 for pad)
            # [batch, 1, 1, seq_len]
            padding_mask = attention_mask[:, None, None, :].bool()
            
            # Final mask: Keep if (Causal AND Padding)
            # [batch, 1, seq_len, seq_len]
            combined_mask = causal_mask & padding_mask
            
            extended_attention_mask = combined_mask

        # --- Handle Position Embeddings (RoPE) ---
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        position_embeddings = None
        if hasattr(base_model, "rotary_emb"):
            # Llama / Qwen2 style
            cos, sin = base_model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # 3. Layer Loop
        # Get layers list
        if hasattr(base_model, "layers"):
            layers = base_model.layers
        elif hasattr(base_model, "h"):
            layers = base_model.h
        else:
            raise ValueError("Unknown layers attribute")
            
        for i, layer in enumerate(layers):
            # layer_mask: [batch, num_layers]
            # mask_i: [batch, 1, 1] for broadcasting
            mask_i = layer_mask[:, i].view(batch_size, 1, 1)
            
            layer_kwargs = {
                "attention_mask": extended_attention_mask,
                "position_ids": position_ids,
            }
            
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            
            # Call layer
            try:
                layer_outputs = layer(hidden_states, **layer_kwargs)
            except TypeError:
                # Fallback
                 layer_outputs = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_ids=position_ids
                )

            # layer_outputs[0] is the hidden state
            new_hidden_states = layer_outputs[0]
            
            # Soft Mixing for Differentiable Pruning (Simulation)
            hidden_states = mask_i * new_hidden_states + (1 - mask_i) * hidden_states
            
        # 4. Final Norm
        if hasattr(base_model, "norm"):
            hidden_states = base_model.norm(hidden_states)
        elif hasattr(base_model, "ln_f"):
            hidden_states = base_model.ln_f(hidden_states)
            
        # 5. LM Head
        if hasattr(self.teacher, "lm_head"):
            logits = self.teacher.lm_head(hidden_states)
        else:
            raise ValueError("Unknown lm_head")
            
        return logits
