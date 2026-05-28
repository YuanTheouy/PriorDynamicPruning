import torch
import torch.nn as nn
import torch.nn.functional as F


class OpalRiskRouter(nn.Module):
    """Predict per-layer skip risk from a compact prompt representation."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0, input_size: int = None):
        super().__init__()
        input_size = int(input_size or hidden_size)
        inner = max(64, input_size // 2)
        self.num_layers = int(num_layers)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, self.num_layers),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).float()


class PromptCandidateMaskRouter(nn.Module):
    """Predict candidate-mask losses from a prompt-only representation."""

    def __init__(self, hidden_size: int, num_candidates: int, dropout: float = 0.0, input_size: int = None):
        super().__init__()
        input_size = int(input_size or hidden_size)
        inner = max(64, input_size // 2)
        self.num_candidates = int(num_candidates)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, self.num_candidates),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).float()


class LayerwiseHiddenRiskRouter(nn.Module):
    """Predict per-layer skip risk from prompt-only hidden states at each layer."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0, input_size: int = None):
        super().__init__()
        input_size = int(input_size or hidden_size)
        inner = max(64, input_size // 2)
        self.num_layers = int(num_layers)
        self.layer_embedding = nn.Parameter(torch.empty(self.num_layers, input_size))
        nn.init.normal_(self.layer_embedding, mean=0.0, std=input_size ** -0.5)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 3:
            raise ValueError(f"LayerwiseHiddenRiskRouter expects [batch, layers, hidden], got {tuple(states.shape)}")
        if states.size(1) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {states.size(1)}")
        x = states.float() + self.layer_embedding.unsqueeze(0)
        return self.net(x).squeeze(-1).float()


class LayerQueryCrossAttentionRiskRouter(nn.Module):
    """Predict per-layer skip risk with learnable layer queries over prompt states."""

    def __init__(
        self,
        base_hidden_size: int,
        num_layers: int,
        router_dim: int = 256,
        router_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.router_dim = int(router_dim)
        self.router_heads = int(router_heads)
        if self.router_dim <= 0:
            raise ValueError("router_dim must be positive")
        if self.router_heads <= 0 or self.router_dim % self.router_heads != 0:
            raise ValueError("router_dim must be divisible by router_heads")

        input_size = int(base_hidden_size) * 2
        self.token_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.router_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.layer_queries = nn.Parameter(torch.empty(self.num_layers, self.router_dim))
        nn.init.normal_(self.layer_queries, mean=0.0, std=self.router_dim ** -0.5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.router_dim,
            num_heads=self.router_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.router_dim),
            nn.Linear(self.router_dim, self.router_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.router_dim, 1),
        )

    def forward(
        self,
        raw_seq: torch.Tensor,
        hk_seq: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_seq = raw_seq.float()
        hk_seq = hk_seq.float()
        x = torch.cat([raw_seq, hk_seq], dim=-1)
        z = self.token_proj(x)
        query = self.layer_queries.unsqueeze(0).expand(z.size(0), -1, -1)
        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        context, _ = self.cross_attn(
            query=query,
            key=z,
            value=z,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.risk_head(context).squeeze(-1).float()


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
