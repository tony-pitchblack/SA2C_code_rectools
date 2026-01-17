from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AlbertConfig, AlbertModel


class Albert4Rec(nn.Module):
    def __init__(
        self,
        *,
        item_num: int,
        state_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        self.item_num = int(item_num)
        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size)

        self.pad_id = 0
        self.mask_id = int(self.item_num) + 1
        self.vocab_size = int(self.item_num) + 2

        cfg = AlbertConfig(
            vocab_size=int(self.vocab_size),
            embedding_size=int(self.hidden_size),
            hidden_size=int(self.hidden_size),
            num_attention_heads=int(num_heads),
            num_hidden_layers=int(num_layers),
            intermediate_size=int(intermediate_size) if intermediate_size is not None else int(self.hidden_size) * 4,
            hidden_dropout_prob=float(dropout_rate),
            attention_probs_dropout_prob=float(dropout_rate),
            max_position_embeddings=int(self.state_size),
            type_vocab_size=1,
        )
        self.albert = AlbertModel(cfg, add_pooling_layer=False)
        self.item_emb = nn.Embedding(int(self.vocab_size), int(self.hidden_size), padding_idx=int(self.pad_id))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if int(seqlen) != int(self.state_size):
            raise ValueError(f"Expected input_ids shape [B,{int(self.state_size)}], got {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.pad_id)).to(torch.long)
        out = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def score_candidates(self, hidden_flat: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(cand_ids)
        logits = (hidden_flat[:, None, :] * emb).sum(dim=-1)
        logits = logits.masked_fill(cand_ids.eq(int(self.pad_id)), float("-inf"))
        logits = logits.masked_fill(cand_ids.eq(int(self.mask_id)), float("-inf"))
        return logits

    def full_item_scores(self, hidden_flat: torch.Tensor) -> torch.Tensor:
        logits = hidden_flat @ self.item_emb.weight.t()
        logits[:, int(self.pad_id)] = float("-inf")
        logits[:, int(self.mask_id)] = float("-inf")
        return logits


__all__ = ["Albert4Rec"]

