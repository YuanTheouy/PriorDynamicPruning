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
            
    def forward(self, input_ids, attention_mask, layer_mask, past_key_values=None, use_cache=False):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            layer_mask: [batch_size, num_layers] - binary mask (0 or 1)
            past_key_values: tuple of KV caches for generation
            use_cache: whether to return past_key_values
        """
        # 直接调用魔改后的 teacher_model 的 forward 方法，传入 layer_mask
        # 由于我们已经在 modeling_qwen2.py 中支持了 layer_mask 参数及 Soft Mixing，
        # 外部不再需要手动提取 embedding, RoPE, 和进行 Layer 循环。
        outputs = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_mask=layer_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False # 推理时不需要隐藏层，节省显存
        )
        
        # 根据是否 use_cache 返回不同结果，适配推理循环
        if use_cache:
            return outputs.logits, outputs.past_key_values
        return outputs.logits
