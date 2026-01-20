from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler

from ..config import resolve_ce_sampling
from ..distributed import get_local_rank, get_world_size, is_distributed, is_rank0
from ..data_utils.sessions import make_session_loader, make_shifted_batch_from_sessions
from ..metrics import evaluate, get_metric_value, ndcg_reward_from_logits
from ..models import SASRecBaselineRectools
from ..utils import tqdm
from .sampling import sample_global_uniform_negatives, sample_uniform_negatives


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
    metric_key: str = "overall.ndcg@10",
    trial=None,
    ce_loss_vocab_size: int | None = None,
    ce_full_vocab_size: int | None = None,
    ce_vocab_pct: float | None = None,
    continue_training: bool = False,
    on_train_log: Callable[[int, dict[str, float]], None] | None = None,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    on_val_end: Callable[[int, dict], None] | None = None,
):
    logger = logging.getLogger(__name__)
    world_size = int(get_world_size())
    if is_distributed():
        num_batches = int(math.ceil(float(num_batches) / float(world_size)))
        if int(max_steps) > 0:
            max_steps = int(math.ceil(float(max_steps) / float(world_size)))
    model = SASRecBaselineRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    if bool(continue_training):
        best_path = run_dir / "best_model.pt"
        if is_rank0():
            logger.info("continue(baseline): enabled")
            logger.info("continue(baseline): run_dir=%s", str(run_dir))
            logger.info("continue(baseline): probing checkpoint=%s", str(best_path))
        if not best_path.exists():
            if is_rank0():
                logger.info("continue(baseline): checkpoint not found")
            raise FileNotFoundError(f"--continue requires {best_path.name} in run_dir={str(run_dir)}")
        state = torch.load(str(best_path), map_location=device)
        model.load_state_dict(state)
        if is_rank0():
            logger.info("continue(baseline): checkpoint loaded successfully")
    if is_distributed():
        local_rank = int(get_local_rank())
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.005)))
    if ce_loss_vocab_size is None or ce_full_vocab_size is None:
        ce_loss_vocab_size, ce_full_vocab_size, ce_vocab_pct, ce_n_neg_eff = resolve_ce_sampling(cfg=cfg, item_num=item_num)
    else:
        _, _, _, ce_n_neg_eff = resolve_ce_sampling(cfg=cfg, item_num=item_num)

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
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_hr10_hits = 0.0
        epoch_ndcg10_sum = 0.0
        epoch_metric_tokens = 0
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
            batch_idx = int(_)
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
            if ce_n_neg_eff is None:
                ce_logits_seq = model(states_x)
                ce_logits = ce_logits_seq[valid_mask]
                loss = F.cross_entropy(ce_logits, action_flat)

                with torch.no_grad():
                    k = int(min(10, int(ce_logits.shape[1])))
                    topk = ce_logits.topk(k=k, dim=1).indices
                    hit = topk.eq(action_flat[:, None]).any(dim=1)
                    epoch_hr10_hits += float(hit.to(torch.float32).sum().item())
                    epoch_ndcg10_sum += float(ndcg_reward_from_logits(ce_logits, action_flat).sum().item())
                    epoch_metric_tokens += int(action_flat.numel())
            else:
                base = model.module if hasattr(model, "module") else model
                seqs = base.encode_seq(states_x)
                seqs_flat = seqs[valid_mask]
                if ce_vocab_pct is not None:
                    neg_ids = sample_global_uniform_negatives(n_neg=int(ce_n_neg_eff), item_num=item_num, device=device)
                    pos_emb = base.item_emb(action_flat)
                    pos_logits = (seqs_flat * pos_emb).sum(dim=-1)
                    neg_emb = base.item_emb(neg_ids)
                    neg_logits = seqs_flat @ neg_emb.t()
                    neg_logits = neg_logits.masked_fill(neg_ids[None, :].eq(action_flat[:, None]), float("-inf"))
                    cand_logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
                    loss = F.cross_entropy(
                        cand_logits,
                        torch.zeros((int(cand_logits.shape[0]),), dtype=torch.long, device=device),
                    )
                else:
                    negs = sample_uniform_negatives(actions=action_flat, n_neg=int(ce_n_neg_eff), item_num=item_num)
                    cand_ids = torch.cat([action_flat[:, None], negs], dim=1)
                    cand_logits = base.score_ce_candidates(seqs_flat, cand_ids)
                    loss = F.cross_entropy(
                        cand_logits,
                        torch.zeros((int(cand_logits.shape[0]),), dtype=torch.long, device=device),
                    )
            if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                raise FloatingPointError(f"Non-finite loss (baseline) at total_step={int(total_step)}")

            n_tok = int(action_flat.numel())
            if n_tok > 0:
                epoch_loss_sum += float(loss.detach().item()) * float(n_tok)
                epoch_tokens += int(n_tok)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_step += int(valid_mask.sum().item())

            if on_train_log is not None and int(n_tok) > 0:
                global_step = int(epoch_idx) * int(max(1, num_batches)) + int(batch_idx + 1)
                on_train_log(int(global_step), {"train_per_batch/loss_ce": float(loss.detach().item())})

        if on_epoch_end is not None and int(epoch_tokens) > 0:
            payload: dict[str, float] = {"train/loss_ce": float(epoch_loss_sum / float(epoch_tokens))}
            if int(epoch_metric_tokens) > 0:
                payload["train/HR_10"] = float(epoch_hr10_hits / float(epoch_metric_tokens))
                payload["train/NDCG_10"] = float(epoch_ndcg10_sum / float(epoch_metric_tokens))
            on_epoch_end(int(epoch_idx + 1), payload)

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
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )
        if on_val_end is not None:
            on_val_end(int(epoch_idx + 1), val_metrics)
        metric = float(get_metric_value(val_metrics, metric_key))
        if trial is not None:
            trial.report(float(metric), step=int(epoch_idx))
            if bool(getattr(trial, "should_prune", lambda: False)()):
                raise RuntimeError("optuna_pruned")
        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            if is_rank0():
                base = model.module if hasattr(model, "module") else model
                torch.save(base.state_dict(), run_dir / "best_model.pt")
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
    if is_rank0() and (not best_path.exists()):
        base = model.module if hasattr(model, "module") else model
        torch.save(base.state_dict(), best_path)
    return best_path


__all__ = ["train_baseline"]

