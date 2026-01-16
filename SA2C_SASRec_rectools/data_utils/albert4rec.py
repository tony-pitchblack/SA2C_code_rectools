from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def _to_albert_tokens(
    items: torch.Tensor,
    is_buy: torch.Tensor,
    *,
    state_size: int,
    purchase_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    items = items.to(torch.long)
    is_buy = is_buy.to(torch.long)
    if bool(purchase_only):
        keep = is_buy.eq(1)
        if bool(keep.any()):
            items = items[keep]
            is_buy = is_buy[keep]
        else:
            items = items[:0]
            is_buy = is_buy[:0]

    s = int(state_size)
    out_items = torch.zeros((s,), dtype=torch.long)
    out_buy = torch.zeros((s,), dtype=torch.long)
    n = int(items.numel())
    if n <= 0:
        return out_items, out_buy
    n = min(int(n), int(s))
    tail_items = items[-n:]
    tail_buy = is_buy[-n:]
    out_items[-n:] = tail_items + 1
    out_buy[-n:] = tail_buy
    return out_items, out_buy


def collate_albert4rec(batch, *, state_size: int, purchase_only: bool):
    items_list, is_buy_list = zip(*batch)
    xs = []
    ys = []
    for items, is_buy in zip(items_list, is_buy_list):
        x, y = _to_albert_tokens(items, is_buy, state_size=int(state_size), purchase_only=bool(purchase_only))
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def make_albert4rec_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    state_size: int,
    purchase_only: bool,
    shuffle: bool,
    sampler=None,
):
    persistent_workers = int(num_workers) > 0
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
        drop_last=False,
        collate_fn=lambda b: collate_albert4rec(b, state_size=int(state_size), purchase_only=bool(purchase_only)),
    )


__all__ = ["collate_albert4rec", "make_albert4rec_loader"]

