import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerRouter(nn.Module):
    def __init__(self, hidden_size, num_layers, top_k, gumbel_noise_scale=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.top_k = top_k
        self.gumbel_noise_scale = float(gumbel_noise_scale)
        
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
        if self.training and self.gumbel_noise_scale > 0:
            # Standard Gumbel(0, 1) noise for top-k exploration.
            uniform = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
            noise = -torch.log(-torch.log(uniform))
            noisy_scores = scores + noise * self.gumbel_noise_scale
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


def mask_distillation_loss(scores, oracle_mask):
    return F.binary_cross_entropy_with_logits(scores.float(), oracle_mask.float())


def risk_ranking_loss(scores, oracle_mask, margin: float = 1.0):
    losses = []
    for row_scores, row_mask in zip(scores.float(), oracle_mask.float()):
        keep_scores = row_scores[row_mask > 0.5]
        skip_scores = row_scores[row_mask <= 0.5]
        if keep_scores.numel() == 0 or skip_scores.numel() == 0:
            continue
        pairwise = margin + skip_scores.view(-1, 1) - keep_scores.view(1, -1)
        losses.append(F.relu(pairwise).mean())
    if not losses:
        return scores.float().new_tensor(0.0)
    return torch.stack(losses).mean()
