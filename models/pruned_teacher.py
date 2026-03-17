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
        # The error indicates that Qwen2's scaled_dot_product_attention expects float/bool mask, but got long.
        # This usually happens when we pass the raw padding mask (0/1 long) directly to SDPA,
        # instead of the prepared 4D mask (float with -inf) or boolean mask.
        
        # `_prepare_decoder_attention_mask` usually returns a 4D float mask (batch, 1, tgt_len, src_len)
        # suited for adding to attention scores (0.0 for keep, min_dtype for mask).
        
        extended_attention_mask = None
        if hasattr(base_model, "_prepare_decoder_attention_mask"):
             extended_attention_mask = base_model._prepare_decoder_attention_mask(
                 attention_mask, 
                 (batch_size, seq_length), 
                 hidden_states, 
                 0 # past_key_values_length
             )
        else:
            # Fallback: Convert to boolean if it's a padding mask (1 for keep, 0 for ignore)
            # Or ensure it's float if it's an additive mask.
            # SDPA usually prefers boolean mask for padding: True to IGNORE (pytorch < 2.0) or True to KEEP?
            # PyTorch SDPA: "attn_mask: boolean mask where a value of True indicates that the element should take part in attention."
            # Our `attention_mask` is 1 for keep, 0 for ignore. So (attention_mask > 0.5) works.
            # But wait, the error says "got attn_mask.dtype: long int".
            
            # If we don't have `_prepare_decoder_attention_mask`, let's try to make it boolean.
            extended_attention_mask = (attention_mask > 0)

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
