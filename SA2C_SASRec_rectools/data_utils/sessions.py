from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SessionDataset(Dataset):
    def __init__(self, data_directory: str, split_df_name: str):
        super().__init__()
        df = pd.read_pickle(os.path.join(data_directory, split_df_name))
        ds = SessionDatasetFromDF(df)
        self.items_list = ds.items_list
        self.is_buy_list = ds.is_buy_list

    def __len__(self):
        return int(len(self.items_list))

    def __getitem__(self, idx: int):
        return self.items_list[int(idx)], self.is_buy_list[int(idx)]


class SessionDatasetFromDF(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        groups = df.groupby("session_id", sort=False)
        items_list = []
        is_buy_list = []
        for _, group in groups:
            items = torch.from_numpy(group["item_id"].to_numpy(dtype=np.int64, copy=True))
            is_buy = torch.from_numpy(group["is_buy"].to_numpy(dtype=np.int64, copy=True))
            if items.numel() == 0:
                continue
            items_list.append(items)
            is_buy_list.append(is_buy)
        self.items_list = items_list
        self.is_buy_list = is_buy_list

    def __len__(self):
        return int(len(self.items_list))

    def __getitem__(self, idx: int):
        return self.items_list[idx], self.is_buy_list[idx]


def collate_sessions(batch, pad_item: int):
    items_list, is_buy_list = zip(*batch)
    lengths = torch.as_tensor([int(x.numel()) for x in items_list], dtype=torch.long)
    lmax = int(lengths.max().item()) if lengths.numel() > 0 else 0
    bsz = int(len(items_list))
    items_pad = torch.full((bsz, lmax), int(pad_item), dtype=torch.long)
    is_buy_pad = torch.zeros((bsz, lmax), dtype=torch.long)
    for i, (items, is_buy) in enumerate(zip(items_list, is_buy_list)):
        n = int(items.numel())
        if n == 0:
            continue
        items_pad[i, :n] = items
        is_buy_pad[i, :n] = is_buy
    return items_pad, is_buy_pad, lengths


def make_session_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    pad_item: int,
    shuffle: bool,
    sampler=None,
):
    num_workers = max(1, int(num_workers))
    persistent_workers = True
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
        drop_last=False,
        collate_fn=lambda b: collate_sessions(b, pad_item=int(pad_item)),
    )


def make_shifted_batch_from_sessions(
    items_pad: torch.Tensor,
    is_buy_pad: torch.Tensor,
    lengths: torch.Tensor,
    *,
    state_size: int,
    old_pad_item: int,
    purchase_only: bool,
):
    bsz, lmax = items_pad.shape
    s = int(state_size)
    if lmax == 0:
        return None

    pos = torch.arange(s, device=items_pad.device).unsqueeze(0).expand(bsz, s)
    base = (lengths - s).unsqueeze(1)
    idx = base + pos
    valid_idx = idx >= 0
    idx = idx.clamp(min=0, max=lmax - 1)

    actions_raw = items_pad.gather(1, idx)
    is_buy_raw = is_buy_pad.gather(1, idx)
    actions_raw = actions_raw.masked_fill(~valid_idx, int(old_pad_item))
    is_buy_raw = is_buy_raw.masked_fill(~valid_idx, 0)

    actions = torch.where(actions_raw == int(old_pad_item), torch.zeros_like(actions_raw), actions_raw + 1).to(
        torch.long
    )
    is_buy = is_buy_raw.to(torch.long)

    states_x = torch.zeros((bsz, s), dtype=torch.long, device=actions.device)
    states_x[:, 1:] = actions[:, :-1]

    valid_mask = actions != 0
    if bool(purchase_only):
        valid_mask = valid_mask & (is_buy == 1)
    if not bool(valid_mask.any()):
        return None

    valid_counts = valid_mask.sum(dim=1)
    keep = valid_counts > 0
    if not bool(keep.all()):
        actions = actions[keep]
        states_x = states_x[keep]
        is_buy = is_buy[keep]
        valid_mask = valid_mask[keep]
        valid_counts = valid_counts[keep]

    last_idx = (valid_counts - 1).clamp(min=0).to(torch.long)
    done_mask = torch.zeros_like(valid_mask)
    done_mask[torch.arange(int(actions.shape[0]), device=actions.device), last_idx] = True

    return {
        "states_x": states_x,
        "actions": actions,
        "is_buy": is_buy,
        "valid_mask": valid_mask,
        "done_mask": done_mask,
    }


__all__ = [
    "SessionDataset",
    "SessionDatasetFromDF",
    "collate_sessions",
    "make_session_loader",
    "make_shifted_batch_from_sessions",
]

