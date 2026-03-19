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
        # 将 layer_mask 挂载到 teacher.config 上，避免修改 forward 的 kwargs 传参
        self.teacher.config.custom_layer_mask = layer_mask
        
        # 为了绝对安全，遍历所有 layer 的 config 进行挂载，防止 config 对象不一致导致掩码丢失
        if hasattr(self.teacher, "model") and hasattr(self.teacher.model, "layers"):
            for layer in self.teacher.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config.custom_layer_mask = layer_mask
        
        # 调用 teacher_model 的 forward 方法
        outputs = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False # 推理时不需要隐藏层，节省显存
        )
        
        # 清理挂载的 mask，防止影响其他前向传播
        self.teacher.config.custom_layer_mask = None
        if hasattr(self.teacher, "model") and hasattr(self.teacher.model, "layers"):
            for layer in self.teacher.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config.custom_layer_mask = None
        
        
        # 根据是否 use_cache 返回不同结果，适配推理循环
        if use_cache:
            return outputs.logits, outputs.past_key_values
        return outputs.logits
