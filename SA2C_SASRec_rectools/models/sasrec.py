from typing import Optional

import torch
import torch.nn as nn


class PointwiseCriticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int], dropout_rate: float):
        super().__init__()
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a non-empty list[int]")
        dims = [int(in_dim)] + [int(x) for x in hidden_sizes] + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if float(dropout_rate) > 0:
                    layers.append(nn.Dropout(float(dropout_rate)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointWiseFeedForward(nn.Module):
    def __init__(self, n_factors: int, n_factors_ff: int, dropout_rate: float):
        super().__init__()
        self.ff_linear_1 = nn.Linear(n_factors, n_factors_ff)
        self.ff_dropout_1 = nn.Dropout(dropout_rate)
        self.ff_activation = nn.ReLU()
        self.ff_linear_2 = nn.Linear(n_factors_ff, n_factors)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        output = self.ff_activation(self.ff_linear_1(seqs))
        return self.ff_linear_2(self.ff_dropout_1(output))


class LearnableInversePositionalEncoding(nn.Module):
    def __init__(self, session_max_len: int, n_factors: int, use_scale_factor: bool = False):
        super().__init__()
        self.pos_emb = nn.Embedding(session_max_len, n_factors)
        self.use_scale_factor = bool(use_scale_factor)

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        bsz, session_max_len, n_factors = sessions.shape
        if self.use_scale_factor:
            sessions = sessions * (n_factors**0.5)
        positions = torch.arange(session_max_len - 1, -1, -1, device=sessions.device)
        sessions = sessions + self.pos_emb(positions)[None, :, :]
        return sessions


class SASRecTransformerLayer(nn.Module):
    def __init__(self, n_factors: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True)
        self.q_layer_norm = nn.LayerNorm(n_factors)
        self.ff_layer_norm = nn.LayerNorm(n_factors)
        self.feed_forward = PointWiseFeedForward(n_factors, n_factors, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        seqs: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.q_layer_norm(seqs)
        mha_output, _ = self.multi_head_attn(
            q, seqs, seqs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        seqs = q + mha_output
        ff_input = self.ff_layer_norm(seqs)
        seqs = self.feed_forward(ff_input)
        seqs = self.dropout(seqs)
        seqs = seqs + ff_input
        return seqs


class SASRecTransformerLayers(nn.Module):
    def __init__(self, n_blocks: int, n_factors: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.transformer_blocks = nn.ModuleList(
            [SASRecTransformerLayer(n_factors, n_heads, dropout_rate) for _ in range(self.n_blocks)]
        )
        self.last_layernorm = nn.LayerNorm(n_factors, eps=1e-8)

    def forward(
        self,
        seqs: torch.Tensor,
        timeline_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for i in range(self.n_blocks):
            seqs = seqs * timeline_mask
            seqs = self.transformer_blocks[i](seqs, attn_mask, key_padding_mask)
        seqs = seqs * timeline_mask
        return self.last_layernorm(seqs)


class SASRecQNetworkRectools(nn.Module):
    def __init__(
        self,
        item_num: int,
        state_size: int,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        *,
        pointwise_critic_use: bool = False,
        pointwise_critic_arch: str = "dot",
        pointwise_critic_mlp: dict | None = None,
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size)
        self.pad_id = 0

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=self.pad_id)
        self.pos_encoding = LearnableInversePositionalEncoding(self.state_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = SASRecTransformerLayers(
            n_blocks=int(num_blocks),
            n_factors=self.hidden_size,
            n_heads=int(num_heads),
            dropout_rate=float(dropout_rate),
        )

        self.head_q = nn.Linear(self.hidden_size, self.item_num + 1)

        self.pointwise_critic_use = bool(pointwise_critic_use)
        self.pointwise_critic_arch = str(pointwise_critic_arch)
        self.pointwise_critic_mlp = None
        if self.pointwise_critic_use:
            if self.pointwise_critic_arch not in {"dot", "mlp"}:
                raise ValueError("pointwise_critic_arch must be one of: dot | mlp")
            if self.pointwise_critic_arch == "mlp":
                mlp_cfg = dict(pointwise_critic_mlp or {})
                hidden_sizes = mlp_cfg.get("hidden_sizes", None)
                dr = mlp_cfg.get("dropout_rate", None)
                if hidden_sizes is None or dr is None:
                    raise ValueError("pointwise_critic_mlp must contain: hidden_sizes, dropout_rate")
                self.pointwise_critic_mlp = PointwiseCriticMLP(
                    in_dim=2 * int(self.hidden_size),
                    hidden_sizes=list(hidden_sizes),
                    dropout_rate=float(dr),
                )

        self.use_causal_attn = bool(use_causal_attn)
        self.use_key_padding_mask = bool(use_key_padding_mask)
        causal = torch.ones(self.state_size, self.state_size, dtype=torch.bool).triu(1)
        self.register_buffer("causal_attn_mask", causal, persistent=False)

    def forward(self, inputs: torch.Tensor, len_state: Optional[torch.Tensor] = None):
        bsz, seqlen = inputs.shape
        if seqlen != self.state_size:
            raise ValueError(f"Expected inputs shape [B,{self.state_size}], got {tuple(inputs.shape)}")

        seqs = self.encode_seq(inputs)
        ce_logits_seq = seqs @ self.item_emb.weight.t()
        ce_logits_seq[:, :, self.pad_id] = float("-inf")
        if self.pointwise_critic_use:
            return ce_logits_seq
        q_values_seq = self.head_q(seqs)
        q_values_seq[:, :, self.pad_id] = float("-inf")
        return q_values_seq, ce_logits_seq

    def encode_seq(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = inputs.shape
        if seqlen != self.state_size:
            raise ValueError(f"Expected inputs shape [B,{self.state_size}], got {tuple(inputs.shape)}")

        timeline_mask = (inputs != self.pad_id).unsqueeze(-1).to(self.item_emb.weight.dtype)

        seqs = self.item_emb(inputs)
        seqs = self.pos_encoding(seqs)
        seqs = self.dropout(seqs)

        attn_mask = None
        key_padding_mask = None
        if self.use_causal_attn:
            attn_mask = self.causal_attn_mask[:seqlen, :seqlen]
        if self.use_key_padding_mask:
            key_padding_mask = inputs == self.pad_id

        seqs = self.layers(seqs, timeline_mask, attn_mask, key_padding_mask)
        return seqs

    def score_ce_candidates(self, seqs_flat: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(cand_ids)
        logits = (seqs_flat[:, None, :] * emb).sum(dim=-1)
        logits = logits.masked_fill(cand_ids.eq(self.pad_id), float("-inf"))
        return logits

    def q_value(self, seqs_flat: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.pointwise_critic_use:
            raise RuntimeError("q_value() is only available when pointwise_critic_use=True")
        if self.pointwise_critic_arch == "mlp":
            if self.pointwise_critic_mlp is None:
                raise RuntimeError("pointwise_critic_mlp is not initialized")
            if item_ids.ndim == 1:
                emb = self.item_emb(item_ids)
                x = torch.cat([seqs_flat, emb], dim=-1)
                q = self.pointwise_critic_mlp(x).squeeze(-1)
                return q.masked_fill(item_ids.eq(self.pad_id), float("-inf"))
            if item_ids.ndim == 2:
                emb = self.item_emb(item_ids)
                n, c, d = emb.shape
                seq_rep = seqs_flat[:, None, :].expand(n, c, d)
                x = torch.cat([seq_rep, emb], dim=-1).reshape(n * c, -1)
                q = self.pointwise_critic_mlp(x).view(n, c)
                return q.masked_fill(item_ids.eq(self.pad_id), float("-inf"))
            raise ValueError(f"Expected item_ids shape [N] or [N,C], got {tuple(item_ids.shape)}")
        if item_ids.ndim == 1:
            emb = self.item_emb(item_ids)
            q = (seqs_flat * emb).sum(dim=-1) + self.head_q.bias[item_ids]
            return q.masked_fill(item_ids.eq(self.pad_id), float("-inf"))
        if item_ids.ndim == 2:
            emb = self.item_emb(item_ids)
            q = (seqs_flat[:, None, :] * emb).sum(dim=-1) + self.head_q.bias[item_ids]
            return q.masked_fill(item_ids.eq(self.pad_id), float("-inf"))
        raise ValueError(f"Expected item_ids shape [N] or [N,C], got {tuple(item_ids.shape)}")

    def score_q_candidates(self, seqs_flat: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        if self.pointwise_critic_use:
            return self.q_value(seqs_flat, cand_ids)
        w = self.head_q.weight[cand_ids]
        b = self.head_q.bias[cand_ids]
        logits = (seqs_flat[:, None, :] * w).sum(dim=-1) + b
        logits = logits.masked_fill(cand_ids.eq(self.pad_id), float("-inf"))
        return logits


class SASRecBaselineRectools(nn.Module):
    def __init__(
        self,
        item_num: int,
        state_size: int,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size)
        self.pad_id = 0

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=self.pad_id)
        self.pos_encoding = LearnableInversePositionalEncoding(self.state_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = SASRecTransformerLayers(
            n_blocks=int(num_blocks),
            n_factors=self.hidden_size,
            n_heads=int(num_heads),
            dropout_rate=float(dropout_rate),
        )

        self.use_causal_attn = bool(use_causal_attn)
        self.use_key_padding_mask = bool(use_key_padding_mask)
        causal = torch.ones(self.state_size, self.state_size, dtype=torch.bool).triu(1)
        self.register_buffer("causal_attn_mask", causal, persistent=False)

    def forward(self, inputs: torch.Tensor, len_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = inputs.shape
        if seqlen != self.state_size:
            raise ValueError(f"Expected inputs shape [B,{self.state_size}], got {tuple(inputs.shape)}")
        seqs = self.encode_seq(inputs)
        ce_logits_seq = seqs @ self.item_emb.weight.t()
        ce_logits_seq[:, :, self.pad_id] = float("-inf")
        return ce_logits_seq

    def encode_seq(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = inputs.shape
        if seqlen != self.state_size:
            raise ValueError(f"Expected inputs shape [B,{self.state_size}], got {tuple(inputs.shape)}")

        timeline_mask = (inputs != self.pad_id).unsqueeze(-1).to(self.item_emb.weight.dtype)

        seqs = self.item_emb(inputs)
        seqs = self.pos_encoding(seqs)
        seqs = self.dropout(seqs)

        attn_mask = None
        key_padding_mask = None
        if self.use_causal_attn:
            attn_mask = self.causal_attn_mask[:seqlen, :seqlen]
        if self.use_key_padding_mask:
            key_padding_mask = inputs == self.pad_id

        seqs = self.layers(seqs, timeline_mask, attn_mask, key_padding_mask)
        return seqs

    def score_ce_candidates(self, seqs_flat: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(cand_ids)
        logits = (seqs_flat[:, None, :] * emb).sum(dim=-1)
        logits = logits.masked_fill(cand_ids.eq(self.pad_id), float("-inf"))
        return logits

