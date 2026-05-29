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
        use_hk_last_residual: bool = False,
        use_raw_last_residual: bool = False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.router_dim = int(router_dim)
        self.router_heads = int(router_heads)
        self.use_hk_last_residual = bool(use_hk_last_residual)
        self.use_raw_last_residual = bool(use_raw_last_residual)
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
        if self.use_hk_last_residual:
            self.hk_last_head = OpalRiskRouter(
                hidden_size=int(base_hidden_size),
                num_layers=self.num_layers,
                dropout=dropout,
            )
        else:
            self.hk_last_head = None
        if self.use_raw_last_residual:
            self.raw_last_head = OpalRiskRouter(
                hidden_size=int(base_hidden_size),
                num_layers=self.num_layers,
                dropout=dropout,
            )
        else:
            self.raw_last_head = None

    @staticmethod
    def _last_active_state(seq: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        rows = []
        for hidden, mask in zip(seq, attention_mask):
            active = mask.ne(0).nonzero(as_tuple=False).flatten()
            rows.append(hidden[active[-1]] if active.numel() > 0 else hidden[-1])
        return torch.stack(rows, dim=0)

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
        risk = self.risk_head(context).squeeze(-1).float()
        if self.hk_last_head is not None:
            risk = risk + self.hk_last_head(self._last_active_state(hk_seq, attention_mask))
        if self.raw_last_head is not None:
            risk = risk + self.raw_last_head(self._last_active_state(raw_seq, attention_mask))
        return risk.float()


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


def exact_k_subset_ce_loss(
    pred_risk: torch.Tensor,
    skip_mask: torch.Tensor,
    allowed_layers: list[int],
    skip_count: int,
    score_clip: float = 50.0,
) -> torch.Tensor:
    """Cross entropy over all allowed cardinality-K skip subsets.

    `pred_risk` is lower-is-better-to-skip, so the subset score uses
    `score = -pred_risk`. The partition function is computed with log-space DP
    instead of enumerating all C(N, K) subsets.
    """
    pred_risk = pred_risk.float()
    skip_mask = skip_mask.float()
    allowed_layers = [int(idx) for idx in allowed_layers]
    skip_count = int(skip_count)
    if pred_risk.dim() != 2:
        raise ValueError(f"pred_risk must be [batch, layers], got {tuple(pred_risk.shape)}")
    if skip_mask.shape != pred_risk.shape:
        raise ValueError(f"skip_mask shape {tuple(skip_mask.shape)} does not match pred_risk {tuple(pred_risk.shape)}")
    if not torch.isfinite(pred_risk).all():
        raise ValueError("pred_risk contains NaN/Inf before exact_k_subset_ce_loss")
    if skip_count <= 0:
        return pred_risk.new_tensor(0.0)
    if not allowed_layers:
        raise ValueError("allowed_layers must be non-empty for exact_k_subset_ce_loss")
    if skip_count > len(allowed_layers):
        raise ValueError(f"skip_count={skip_count} exceeds allowed layer count={len(allowed_layers)}")

    allowed = torch.tensor(allowed_layers, dtype=torch.long, device=pred_risk.device)
    scores = -pred_risk.index_select(dim=1, index=allowed)
    if score_clip and float(score_clip) > 0:
        scores = scores.clamp(min=-float(score_clip), max=float(score_clip))
    target = skip_mask.index_select(dim=1, index=allowed)
    target_counts = target.sum(dim=1)
    if not torch.allclose(target_counts, torch.full_like(target_counts, float(skip_count))):
        raise ValueError(
            "Each skip_mask row must contain exactly skip_count skipped layers inside allowed_layers; "
            f"got counts={target_counts.detach().cpu().tolist()} skip_count={skip_count}"
        )

    losses = []
    for row_scores, row_target in zip(scores, target):
        dp = [row_scores.new_tensor(0.0)] + [None for _ in range(skip_count)]
        seen = 0
        for score in row_scores:
            upper = min(skip_count, seen + 1)
            for j in range(upper, 0, -1):
                include_score = dp[j - 1] + score
                dp[j] = include_score if dp[j] is None else torch.logaddexp(dp[j], include_score)
            seen += 1
        teacher_score = row_scores[row_target > 0.5].sum()
        losses.append(dp[skip_count] - teacher_score)
    return torch.stack(losses).mean()
