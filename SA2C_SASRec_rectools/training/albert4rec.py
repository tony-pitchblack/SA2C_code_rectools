from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from ..data_utils.albert4rec import make_albert4rec_loader
from ..metrics import evaluate_albert4rec_loo, get_metric_value
from ..models.albert4rec import Albert4Rec
from ..utils import tqdm


def _mask_inputs(
    input_ids: torch.Tensor,
    *,
    masking_proba: float,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape [B,S], got {tuple(input_ids.shape)}")
    nonpad = input_ids.ne(0)
    if not bool(nonpad.any()):
        masked = input_ids.clone()
        return masked, torch.empty((0,), dtype=torch.long, device=input_ids.device), torch.zeros_like(input_ids, dtype=torch.bool)

    r = torch.rand(input_ids.shape, device=input_ids.device)
    mask = nonpad & (r < float(masking_proba))

    lengths = nonpad.sum(dim=1).to(torch.long)
    has = mask.any(dim=1)
    need = (~has) & lengths.gt(0)
    if bool(need.any()):
        last = (lengths - 1).clamp(min=0)
        rows = torch.where(need)[0]
        mask[rows, last[rows]] = True

    pos = input_ids[mask].to(torch.long)
    masked = input_ids.clone()
    masked[mask] = int(mask_id)
    return masked, pos, mask


def _sample_negatives(pos_ids: torch.Tensor, *, n_negatives: int, item_num: int) -> torch.Tensor:
    pos_ids = pos_ids.to(torch.long)
    n = int(pos_ids.shape[0])
    neg = int(n_negatives)
    if neg <= 0 or n <= 0:
        return torch.empty((n, 0), dtype=torch.long, device=pos_ids.device)
    neg_ids = torch.randint(1, int(item_num) + 1, size=(n, neg), device=pos_ids.device, dtype=torch.long)
    bad = neg_ids.eq(pos_ids[:, None])
    while bool(bad.any()):
        k = int(bad.sum().item())
        neg_ids[bad] = torch.randint(1, int(item_num) + 1, size=(k,), device=pos_ids.device, dtype=torch.long)
        bad = neg_ids.eq(pos_ids[:, None])
    return neg_ids


def train_albert4rec(
    *,
    cfg: dict,
    train_ds,
    val_dl,
    run_dir: Path,
    device: torch.device,
    reward_click: float,
    reward_buy: float,
    state_size: int,
    item_num: int,
    purchase_only: bool,
    num_epochs: int,
    num_batches: int,
    train_batch_size: int,
    train_num_workers: int,
    pin_memory: bool,
    max_steps: int,
    metric_key: str = "overall.ndcg@10",
):
    logger = logging.getLogger(__name__)
    a4 = cfg.get("albert4rec") or {}
    if not isinstance(a4, dict):
        raise ValueError("albert4rec must be a mapping (dict)")
    masking_proba = float(a4.get("masking_proba", 0.2))
    n_negatives = int(a4.get("n_negatives", 256))
    intermediate_size = a4.get("intermediate_size", None)
    if intermediate_size is not None:
        intermediate_size = int(intermediate_size)

    model = Albert4Rec(
        item_num=int(item_num),
        state_size=int(state_size),
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_layers=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        intermediate_size=intermediate_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-4)))

    total_step = 0
    early_patience = int(cfg.get("early_stopping_ep", 5))
    best_metric = float("-inf")
    epochs_since_improve = 0
    stop_training = False

    for epoch_idx in range(num_epochs):
        if num_batches > 0:
            sampler = RandomSampler(train_ds, replacement=True, num_samples=num_batches * int(train_batch_size))
            t0 = time.perf_counter()
            dl = make_albert4rec_loader(
                train_ds,
                batch_size=train_batch_size,
                num_workers=train_num_workers,
                pin_memory=pin_memory,
                state_size=int(state_size),
                purchase_only=bool(purchase_only),
                shuffle=False,
                sampler=sampler,
            )
            train_dl_s = time.perf_counter() - t0
        else:
            dl = []
            train_dl_s = 0.0
        if epoch_idx == 0:
            logger.info("build_s train_dl=%.3f", float(train_dl_s))

        model.train()
        for batch in tqdm(
            dl,
            total=num_batches,
            desc=f"train epoch {epoch_idx + 1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
        ):
            if max_steps > 0 and total_step >= max_steps:
                stop_training = True
                break

            input_ids, _is_buy = batch
            input_ids = input_ids.to(device, non_blocking=pin_memory).to(torch.long)
            masked, pos, mask = _mask_inputs(input_ids, masking_proba=float(masking_proba), mask_id=int(model.mask_id))
            if int(pos.numel()) <= 0:
                continue

            h = model(masked)
            h_flat = h[mask]
            pos = pos.to(device, non_blocking=pin_memory)
            neg = _sample_negatives(pos, n_negatives=int(n_negatives), item_num=int(item_num))
            cand = torch.cat([pos[:, None], neg], dim=1)
            logits = model.score_candidates(h_flat, cand)
            loss = F.cross_entropy(logits, torch.zeros((int(pos.shape[0]),), dtype=torch.long, device=device))
            if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                raise FloatingPointError(f"Non-finite loss (albert4rec) at total_step={int(total_step)}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_step += int(pos.shape[0])

        val_metrics = evaluate_albert4rec_loo(
            model,
            val_dl,
            reward_click,
            reward_buy,
            device,
            debug=bool(cfg.get("debug", False)),
            split="val",
            state_size=state_size,
            item_num=item_num,
            purchase_only=purchase_only,
            epoch=int(epoch_idx + 1),
            num_epochs=int(num_epochs),
        )
        metric = float(get_metric_value(val_metrics, metric_key))
        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            logger.info("best_model.pt updated (val %s=%f)", str(metric_key), float(best_metric))
        else:
            epochs_since_improve += 1
            logger.info(
                "no improvement (val %s=%f best=%f) patience=%d/%d",
                str(metric_key),
                float(metric),
                float(best_metric),
                int(epochs_since_improve),
                int(early_patience),
            )
            if early_patience > 0 and epochs_since_improve >= early_patience:
                logger.info("early stopping triggered")
                break
        if stop_training:
            logger.info("max_steps reached; stopping")
            break

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    return best_path


__all__ = ["train_albert4rec"]

