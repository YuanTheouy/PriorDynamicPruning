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
        # Qwen2 and Llama use Rotary Embeddings, which are usually computed inside the model
        # and passed to layers as `position_embeddings` (cos, sin) tuple.
        # If we just call layer(hidden_states, attention_mask=...), it might fail if it expects position_embeddings.
        
        # We need to replicate the preparation logic from the base model's forward.
        # This includes:
        # - Creating the extended attention mask
        # - Computing rotary embeddings (cos, sin)
        
        # --- Handle Attention Mask ---
        # Try to use internal helper if available
        extended_attention_mask = None
        if hasattr(base_model, "_prepare_decoder_attention_mask"):
             extended_attention_mask = base_model._prepare_decoder_attention_mask(
                 attention_mask, 
                 (batch_size, seq_length), 
                 hidden_states, 
                 0 # past_key_values_length
             )
        else:
            # Fallback or simple mask (might be insufficient for some models)
            extended_attention_mask = attention_mask

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
            
            # Prepare arguments for layer forward
            # Different models have different signatures.
            # Qwen2/Llama: (hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs)
            # OR (hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None)
            
            # The error `TypeError: cannot unpack non-iterable NoneType object` at `cos, sin = position_embeddings`
            # suggests that `position_embeddings` was passed as None, or it wasn't passed and the layer tried to compute/unpack it but failed.
            
            # Let's try to pass arguments as kwargs to be safe, or explicit if we know them.
            # For Qwen2/Llama, passing `position_embeddings` explicitly is often required if not computed inside layer.
            
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
                # Fallback: maybe it doesn't accept position_embeddings directly?
                # Or maybe it expects it as positional args?
                # Let's try minimal args
                 layer_outputs = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_ids=position_ids
                )

            # layer_outputs[0] is the hidden state
            new_hidden_states = layer_outputs[0]
            
            # Soft Mixing for Differentiable Pruning (Simulation)
            # If mask_i is 1, keep new; if 0, keep old (skip)
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
