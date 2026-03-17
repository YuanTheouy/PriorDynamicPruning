import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

class OneLayerStudentModel(nn.Module):
    def __init__(self, teacher_model_name_or_path, sid_token_ids):
        super().__init__()
        # 1. Load Teacher Config and modify to 1 layer
        config = AutoConfig.from_pretrained(teacher_model_name_or_path)
        config.num_hidden_layers = 1
        
        # 2. Initialize a single-layer base model
        # We only need the backbone (without the full LM Head)
        # Using from_config creates randomly initialized weights
        causal_model = AutoModelForCausalLM.from_config(config)
        self.backbone = causal_model.model 
        
        # 3. Build our streamlined LM Head
        hidden_size = config.hidden_size
        vocab_size = len(sid_token_ids)
        self.sid_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 4. Store aux info
        self.sid_token_ids = sid_token_ids
        
    def init_from_teacher(self, teacher_model):
        """Copy parameters from the teacher model"""
        print("Initializing student from teacher...")
        
        # a. Copy Embedding Layer
        self.backbone.embed_tokens.load_state_dict(teacher_model.model.embed_tokens.state_dict())
        print("  - Copied embed_tokens")
        
        # b. Copy the first Transformer Block
        self.backbone.layers[0].load_state_dict(teacher_model.model.layers[0].state_dict())
        print("  - Copied layer 0")
        
        # c. Copy the final LayerNorm
        self.backbone.norm.load_state_dict(teacher_model.model.norm.state_dict())
        print("  - Copied final norm")
        
        # d. Copy specific SID rows from Teacher's LM Head
        with torch.no_grad():
            self.sid_lm_head.weight.copy_(teacher_model.lm_head.weight[self.sid_token_ids])
        print(f"  - Copied {len(self.sid_token_ids)} rows into sid_lm_head")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # 1. Forward through Backbone
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True,
            **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Forward through the streamlined LM Head
        # logits shape: [batch_size, seq_len, len(sid_token_ids)]
        logits_sid = self.sid_lm_head(last_hidden_state)
        
        return {
            "logits_sid": logits_sid,
            "last_hidden_state": last_hidden_state
        }
