import torch
import torch.nn as nn
import torch.nn.functional as F


class OpalRiskRouter(nn.Module):
    """Predict per-layer skip risk from a pooled prompt representation."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        inner = max(64, int(hidden_size) // 2)
        self.num_layers = int(num_layers)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, self.num_layers),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).float()


def risk_regression_loss(pred_risk: torch.Tensor, target_risk: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred_risk.float(), target_risk.float(), beta=float(beta))


def risk_pairwise_ranking_loss(pred_risk: torch.Tensor, target_risk: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    pred_diff = pred_risk.float().unsqueeze(2) - pred_risk.float().unsqueeze(1)
    target_diff = target_risk.float().unsqueeze(2) - target_risk.float().unsqueeze(1)
    order = target_diff.sign()
    valid = order.ne(0)
    if not valid.any():
        return pred_risk.float().new_tensor(0.0)
    # If target_i > target_j, pred_i should also be greater than pred_j.
    losses = F.softplus(-(pred_diff - float(margin) * order) * order)
    return losses[valid].mean()


def risk_skip_set_loss(pred_risk: torch.Tensor, target_risk: torch.Tensor, skip_count: int) -> torch.Tensor:
    """Train the decision boundary used at inference: the lowest-risk layers are skipped."""
    skip_count = max(0, min(int(skip_count), int(target_risk.size(-1))))
    if skip_count <= 0:
        return pred_risk.float().new_tensor(0.0)
    skip_targets = torch.zeros_like(target_risk.float())
    skip_indices = torch.topk(target_risk.float(), k=skip_count, dim=-1, largest=False).indices
    skip_targets.scatter_(dim=-1, index=skip_indices, value=1.0)
    return F.binary_cross_entropy_with_logits(-pred_risk.float(), skip_targets)
