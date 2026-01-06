import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff1 = nn.Linear(hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        y = self.ln1(x)
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        attn_out, _ = self.attn(
            y,
            y,
            y,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)
        y = self.ln2(x)
        y = self.ff2(self.drop2(F.relu(self.ff1(y))))
        x = x + self.drop2(y)
        return x


class SASRecQNetworkTorch(nn.Module):
    def __init__(self, item_num, state_size, hidden_size, num_heads, num_blocks, dropout_rate):
        super().__init__()
        self.item_num = int(item_num)
        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size)
        self.pad_id = self.item_num

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.state_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [SASRecBlock(self.hidden_size, num_heads=num_heads, dropout=dropout_rate) for _ in range(num_blocks)]
        )
        self.final_ln = nn.LayerNorm(self.hidden_size)
        self.head_q = nn.Linear(self.hidden_size, self.item_num)
        self.head_ce = nn.Linear(self.hidden_size, self.item_num)

        causal = torch.ones(self.state_size, self.state_size, dtype=torch.bool).triu(1)
        self.register_buffer("causal_attn_mask", causal, persistent=False)

    def forward(self, inputs, len_state):
        bsz, seqlen = inputs.shape
        if seqlen != self.state_size:
            raise ValueError(f"Expected inputs shape [B,{self.state_size}], got {tuple(inputs.shape)}")

        positions = torch.arange(self.state_size, device=inputs.device)
        x = self.item_emb(inputs) + self.pos_emb(positions)[None, :, :]
        x = self.dropout(x)
        seq_mask = (inputs != self.pad_id).unsqueeze(-1).to(x.dtype)
        x = x * seq_mask

        key_padding_mask = inputs == self.pad_id
        if key_padding_mask.any():
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        attn_mask = self.causal_attn_mask[:seqlen, :seqlen]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = x * seq_mask

        x = self.final_ln(x)
        idx = (len_state.to(inputs.device) - 1).clamp(min=0)
        pooled = x[torch.arange(bsz, device=inputs.device), idx]
        q_values = self.head_q(pooled)
        ce_logits = self.head_ce(pooled)
        return q_values, ce_logits


