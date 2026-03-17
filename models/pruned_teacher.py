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
        # Access the base model (e.g. LlamaModel)
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

        # 2. Iterative Layers with Masking
        # Get layers list
        if hasattr(base_model, "layers"):
            layers = base_model.layers
        elif hasattr(base_model, "h"):
            layers = base_model.h
        else:
            raise ValueError("Unknown layers attribute")

        # Prepare causal mask (standard for decoder-only)
        # We rely on the model's internal prepare_inputs_for_generation or similar?
        # Actually, if we call layer() directly, we might need to handle the attention mask format.
        # However, calling layer(hidden_states, attention_mask=...) usually expects the expanded mask (batch, 1, 1, seq_len) or similar.
        # To avoid re-implementing the complex mask preparation logic of HF, 
        # we can try to use the model's _prepare_decoder_attention_mask if available,
        # or just rely on the fact that 'attention_mask' passed here is usually the padding mask.
        
        # NOTE: LlamaModel.forward does mask expansion. We should replicate it or use a helper.
        # To be safe and simple: use the base_model's _prepare_decoder_attention_mask if it exists.
        
        # Let's try to get the extended attention mask
        # This is tricky because it's usually private or protected.
        # Plan B: Let's assume the standard `attention_mask` (batch, seq) works if we pass it 
        # but LlamaLayer usually expects 4D mask.
        
        # Let's inspect how to get the prepared mask.
        # We can cheat: run a dummy forward on the base model to get the prepared mask? No, that's slow.
        # Let's try to use `base_model._prepare_decoder_attention_mask`
        
        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0
        
        # Creating the 4D mask manually if needed is safer than relying on private methods
        # But different models have different requirements.
        # Let's look at the `train_student_distill.py` imports. It imports `AutoModelForCausalLM`.
        
        # OPTIMIZATION:
        # Instead of re-implementing forward, we can use a hook or just carefully call layers.
        # For Llama, the attention_mask in forward is expanded.
        
        # Let's reuse `base_model._prepare_decoder_attention_mask` if possible.
        # It takes (attention_mask, input_shape, inputs_embeds, past_key_values_length)
        
        extended_attention_mask = None
        if hasattr(base_model, "_prepare_decoder_attention_mask"):
             extended_attention_mask = base_model._prepare_decoder_attention_mask(
                 attention_mask, 
                 (batch_size, seq_length), 
                 hidden_states, 
                 past_key_values_length
             )
        
        # 3. Layer Loop
        for i, layer in enumerate(layers):
            # layer_mask: [batch, num_layers]
            # mask_i: [batch, 1, 1] for broadcasting
            mask_i = layer_mask[:, i].view(batch_size, 1, 1)
            
            # Compute layer output
            # We assume layer signature is (hidden_states, attention_mask=...)
            # We use the extended mask if available, else the raw one
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask if extended_attention_mask is not None else attention_mask,
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

