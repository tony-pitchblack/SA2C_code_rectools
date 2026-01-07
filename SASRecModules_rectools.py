import torch
import torch.nn as nn
from typing import Optional


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

        self.use_causal_attn = bool(use_causal_attn)
        self.use_key_padding_mask = bool(use_key_padding_mask)
        causal = torch.ones(self.state_size, self.state_size, dtype=torch.bool).triu(1)
        self.register_buffer("causal_attn_mask", causal, persistent=False)

    def forward(self, inputs: torch.Tensor, len_state: torch.Tensor):
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

        has_any = (inputs != self.pad_id).any(dim=1)
        last_pos = torch.full((bsz,), self.state_size - 1, device=inputs.device, dtype=torch.long)
        idx = torch.where(has_any, last_pos, torch.zeros_like(last_pos))
        pooled = seqs[torch.arange(bsz, device=inputs.device), idx]

        ce_logits = pooled @ self.item_emb.weight.t()
        q_values = self.head_q(pooled)
        q_values[:, self.pad_id] = float("-inf")
        ce_logits[:, self.pad_id] = float("-inf")
        return q_values, ce_logits


