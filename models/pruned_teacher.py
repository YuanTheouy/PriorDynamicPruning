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
        # --- NEW APPROACH: Rely on HuggingFace's internal machinery ---
        # Instead of manually extracting embeddings, preparing complex 4D masks,
        # computing RoPE, and iterating over layers (which is extremely fragile 
        # and prone to subtle bugs across different model versions),
        # we can use PyTorch hooks to conditionally skip layers during a standard forward pass!
        
        batch_size, num_layers = layer_mask.shape
        
        # Determine the base model
        if hasattr(self.teacher, "model"):
            base_model = self.teacher.model
        elif hasattr(self.teacher, "transformer"):
            base_model = self.teacher.transformer
        else:
            raise ValueError("Unknown teacher model structure")
            
        # Get layers
        if hasattr(base_model, "layers"):
            layers = base_model.layers
        elif hasattr(base_model, "h"):
            layers = base_model.h
        else:
            raise ValueError("Unknown layers attribute")

        # Create forward pre-hooks to implement dynamic pruning
        hooks = []
        
        # We need a way to pass the mask value to the layer.
        # However, hooks cannot easily "skip" the execution of the original module
        # without raising an exception or returning a dummy value (which breaks the computational graph).
        # Wait, if we return `hidden_states` directly from a hook, it might not work easily because 
        # a standard hook doesn't skip the layer execution unless we do some tricky stuff.
        
        # Let's use a Monkey Patching approach for the duration of this forward pass.
        # We will wrap the forward method of each layer.
        
        original_forwards = {}
        
        for i, layer in enumerate(layers):
            original_forwards[i] = layer.forward
            
            # Create a bound method that captures 'i' and 'layer'
            def create_pruned_forward(layer_idx, orig_forward):
                def pruned_forward(*args, **kwargs):
                    # args[0] is typically hidden_states
                    hidden_states = args[0]
                    
                    # mask_i shape: [batch, 1, 1]
                    mask_i = layer_mask[:, layer_idx].view(batch_size, 1, 1)
                    
                    # Call the original forward to get the new hidden states
                    # (We must compute it for Soft Masking to allow gradients to flow back to Router)
                    # For inference (binary mask), if mask is all 0, we could theoretically skip, 
                    # but since batch items might have different masks, we have to compute it anyway.
                    layer_outputs = orig_forward(*args, **kwargs)
                    
                    # layer_outputs is a tuple, layer_outputs[0] is the new hidden_states
                    new_hidden_states = layer_outputs[0]
                    
                    # Soft Mixing
                    mixed_hidden_states = mask_i * new_hidden_states + (1 - mask_i) * hidden_states
                    
                    # Return tuple with modified hidden states, keep other outputs (e.g. past_key_values) intact
                    return (mixed_hidden_states,) + layer_outputs[1:]
                    
                return pruned_forward
                
            # Apply monkey patch
            layer.forward = create_pruned_forward(i, original_forwards[i])
            
        try:
            # Now run the standard HuggingFace forward pass!
            # This automatically handles all the complex logic:
            # - Token Embeddings
            # - Attention Mask Expansion (4D, causal, padding, etc.)
            # - RoPE (Position Embeddings)
            # - KV Cache passing
            # - Final LayerNorm
            # - LM Head
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False # Disable cache during pruning simulation to avoid complex KV mixing
            )
            logits = outputs.logits
        finally:
            # MUST Restore original forward methods
            for i, layer in enumerate(layers):
                layer.forward = original_forwards[i]

        return logits
