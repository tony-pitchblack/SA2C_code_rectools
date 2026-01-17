from __future__ import annotations

import torch


def sample_uniform_negatives(*, actions: torch.Tensor, n_neg: int, item_num: int) -> torch.Tensor:
    actions = actions.to(torch.long)
    n = int(actions.shape[0])
    n_neg = int(n_neg)
    if n_neg <= 0:
        return torch.empty((n, 0), dtype=torch.long, device=actions.device)

    min_id = 1
    max_id_exclusive = int(item_num) + 1
    neg_actions = torch.randint(int(min_id), int(max_id_exclusive), size=(n, n_neg), device=actions.device).to(torch.long)
    bad = neg_actions.eq(actions[:, None])
    while bool(bad.any()):
        neg_actions[bad] = torch.randint(
            int(min_id),
            int(max_id_exclusive),
            size=(int(bad.sum().item()),),
            device=actions.device,
        ).to(torch.long)
        bad = neg_actions.eq(actions[:, None])
    return neg_actions


def sample_global_uniform_negatives(*, n_neg: int, item_num: int, device: torch.device) -> torch.Tensor:
    n_neg = int(n_neg)
    if n_neg <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    min_id = 1
    max_id_exclusive = int(item_num) + 1
    return torch.randint(int(min_id), int(max_id_exclusive), size=(int(n_neg),), device=device, dtype=torch.long)


__all__ = ["sample_uniform_negatives", "sample_global_uniform_negatives"]

