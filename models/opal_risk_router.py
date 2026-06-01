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
        use_layer_self_attention: bool = False,
        budget_condition: str = "none",
        max_budget: int = None,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.router_dim = int(router_dim)
        self.router_heads = int(router_heads)
        self.use_hk_last_residual = bool(use_hk_last_residual)
        self.use_raw_last_residual = bool(use_raw_last_residual)
        self.use_layer_self_attention = bool(use_layer_self_attention)
        self.budget_condition = str(budget_condition or "none")
        self.max_budget = int(max_budget or self.num_layers)
        if self.router_dim <= 0:
            raise ValueError("router_dim must be positive")
        if self.router_heads <= 0 or self.router_dim % self.router_heads != 0:
            raise ValueError("router_dim must be divisible by router_heads")
        if self.budget_condition not in {"none", "skip_count", "keep_count"}:
            raise ValueError("budget_condition must be one of: none, skip_count, keep_count")
        if self.max_budget < 0:
            raise ValueError("max_budget must be non-negative")

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
        if self.use_layer_self_attention:
            self.layer_self_attn_norm = nn.LayerNorm(self.router_dim)
            self.layer_self_attn = nn.MultiheadAttention(
                embed_dim=self.router_dim,
                num_heads=self.router_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.layer_self_ffn = nn.Sequential(
                nn.LayerNorm(self.router_dim),
                nn.Linear(self.router_dim, self.router_dim * 4),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.router_dim * 4, self.router_dim),
                nn.Dropout(float(dropout)),
            )
        else:
            self.layer_self_attn_norm = None
            self.layer_self_attn = None
            self.layer_self_ffn = None
        if self.budget_condition == "none":
            self.budget_embedding = None
        else:
            self.budget_embedding = nn.Embedding(self.max_budget + 1, self.router_dim)
            nn.init.normal_(self.budget_embedding.weight, mean=0.0, std=self.router_dim ** -0.5)
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

    def _budget_values(
        self,
        batch_size: int,
        device: torch.device,
        skip_count=None,
        keep_count=None,
    ):
        if self.budget_condition == "none":
            return None
        value = skip_count if self.budget_condition == "skip_count" else keep_count
        if value is None:
            raise ValueError(f"budget_condition={self.budget_condition!r} requires a budget value in forward()")
        if isinstance(value, torch.Tensor):
            budget = value.to(device=device, dtype=torch.long).flatten()
        elif isinstance(value, (list, tuple)):
            budget = torch.tensor(value, device=device, dtype=torch.long).flatten()
        else:
            budget = torch.full((batch_size,), int(value), device=device, dtype=torch.long)
        if budget.numel() == 1:
            budget = budget.expand(batch_size)
        if budget.numel() != batch_size:
            raise ValueError(f"Budget value must be scalar or batch-sized; got {budget.numel()} for batch={batch_size}")
        if budget.min().item() < 0 or budget.max().item() > self.max_budget:
            raise ValueError(
                f"Budget values must be in [0, {self.max_budget}] for {self.budget_condition}; "
                f"got min={int(budget.min().item())} max={int(budget.max().item())}"
            )
        return budget

    def forward(
        self,
        raw_seq: torch.Tensor,
        hk_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        skip_count=None,
        keep_count=None,
    ) -> torch.Tensor:
        raw_seq = raw_seq.float()
        hk_seq = hk_seq.float()
        x = torch.cat([raw_seq, hk_seq], dim=-1)
        z = self.token_proj(x)
        query = self.layer_queries.unsqueeze(0).expand(z.size(0), -1, -1)
        budget_values = self._budget_values(
            batch_size=z.size(0),
            device=z.device,
            skip_count=skip_count,
            keep_count=keep_count,
        )
        if budget_values is not None:
            query = query + self.budget_embedding(budget_values).unsqueeze(1)
        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        context, _ = self.cross_attn(
            query=query,
            key=z,
            value=z,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if self.layer_self_attn is not None:
            attn_input = self.layer_self_attn_norm(context)
            delta, _ = self.layer_self_attn(
                query=attn_input,
                key=attn_input,
                value=attn_input,
                need_weights=False,
            )
            context = context + delta
            context = context + self.layer_self_ffn(context)
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
