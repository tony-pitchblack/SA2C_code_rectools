from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from ..data_utils.sessions import make_session_loader, make_shifted_batch_from_sessions
from ..metrics import evaluate
from ..models import SASRecBaselineRectools
from ..utils import tqdm


def train_baseline(
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
    evaluate_fn=None,
):
    logger = logging.getLogger(__name__)
    model = SASRecBaselineRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.005)))

    total_step = 0
    early_patience = int(cfg.get("early_stopping_ep", 5))
    best_metric = float("-inf")
    epochs_since_improve = 0
    stop_training = False

    train_ds_s = 0.0
    val_ds_s = 0.0
    test_ds_s = 0.0
    val_dl_s = 0.0
    test_dl_s = 0.0

    for epoch_idx in range(num_epochs):
        if num_batches > 0:
            sampler = RandomSampler(train_ds, replacement=True, num_samples=num_batches * int(train_batch_size))
            t0 = time.perf_counter()
            dl = make_session_loader(
                train_ds,
                batch_size=train_batch_size,
                num_workers=train_num_workers,
                pin_memory=pin_memory,
                pad_item=item_num,
                shuffle=False,
                sampler=sampler,
            )
            train_dl_s = time.perf_counter() - t0
        else:
            dl = []
            train_dl_s = 0.0
        if epoch_idx == 0:
            logger.info(
                "build_s train_ds=%.3f train_dl=%.3f val_ds=%.3f val_dl=%.3f test_ds=%.3f test_dl=%.3f",
                float(train_ds_s),
                float(train_dl_s),
                float(val_ds_s),
                float(val_dl_s),
                float(test_ds_s),
                float(test_dl_s),
            )

        model.train()
        for _, batch in enumerate(
            tqdm(
                dl,
                total=num_batches,
                desc=f"train epoch {epoch_idx + 1}/{num_epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
        ):
            if max_steps > 0 and total_step >= max_steps:
                stop_training = True
                break

            items_pad, is_buy_pad, lengths = batch
            step = make_shifted_batch_from_sessions(
                items_pad,
                is_buy_pad,
                lengths,
                state_size=int(state_size),
                old_pad_item=int(item_num),
                purchase_only=bool(purchase_only),
            )
            if step is None:
                continue

            states_x = step["states_x"].to(device, non_blocking=pin_memory)
            actions = step["actions"].to(device, non_blocking=pin_memory).to(torch.long)
            valid_mask = step["valid_mask"].to(device, non_blocking=pin_memory)

            action_flat = actions[valid_mask]
            ce_logits_seq = model(states_x)
            ce_logits = ce_logits_seq[valid_mask]
            loss = F.cross_entropy(ce_logits, action_flat)
            if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                raise FloatingPointError(f"Non-finite loss (baseline) at total_step={int(total_step)}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_step += int(valid_mask.sum().item())

        eval_fn = evaluate if evaluate_fn is None else evaluate_fn
        val_metrics = eval_fn(
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
        metric = float(val_metrics["overall"].get("ndcg@10", 0.0))
        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            logger.info("best_model.pt updated (val ndcg@10=%f)", float(best_metric))
        else:
            epochs_since_improve += 1
            logger.info(
                "no improvement (val ndcg@10=%f best=%f) patience=%d/%d",
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


__all__ = ["train_baseline"]

