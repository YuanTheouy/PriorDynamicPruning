import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerRouter(nn.Module):
    def __init__(self, hidden_size, num_layers, top_k):
        super().__init__()
        self.num_layers = num_layers
        self.top_k = top_k
        
        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_layers)
        )
        
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: [batch_size, hidden_size] - typically the last token's hidden state
        Returns:
            mask: [batch_size, num_layers] - binary mask with gradient (via STE)
            scores: [batch_size, num_layers] - raw scores
        """
        scores = self.mlp(hidden_state) # [batch_size, num_layers]
        
        # Top-K Selection
        # We want to select the top k layers to KEEP
        # To avoid Mode Collapse (where unselected layers get 0 gradient),
        # we inject Gumbel noise during training to encourage exploration.
        if self.training:
            # Gumbel(0, 1) noise
            noise = -torch.empty_like(scores).exponential_(1e-5).log()
            # Add noise to scores (temperature can be adjusted)
            noisy_scores = scores + noise * 1.0 
            topk_values, topk_indices = torch.topk(noisy_scores, self.top_k, dim=-1)
        else:
            # Deterministic for inference
            topk_values, topk_indices = torch.topk(scores, self.top_k, dim=-1)
            
        # Create Hard Mask
        mask_hard = torch.zeros_like(scores)
        mask_hard.scatter_(-1, topk_indices, 1.0)
        
        # Straight-Through Estimator (STE)
        # Forward pass: uses mask_hard (binary)
        # Backward pass: gradients flow through scores (using sigmoid to bound the gradient scale)
        mask = (mask_hard - torch.sigmoid(scores)).detach() + torch.sigmoid(scores)
        
        return mask, scores
