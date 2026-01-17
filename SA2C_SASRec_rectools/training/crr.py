from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler

from ..config import validate_pointwise_critic_cfg
from ..distributed import get_local_rank, get_world_size, is_distributed, is_rank0
from ..data_utils.sessions import make_session_loader, make_shifted_batch_from_sessions
from ..metrics import evaluate, get_metric_value, ndcg_reward_from_logits
from ..models import SASRecQNetworkRectools
from ..utils import tqdm


def _soft_update_(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    t = float(tau)
    if not (0.0 < t <= 1.0):
        raise ValueError("crr.tau must be in (0, 1]")
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1.0 - t).add_(p.data, alpha=t)


def train_crr(
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
    reward_fn: str,
    evaluate_fn=None,
    metric_key: str = "overall.ndcg@10",
    trial=None,
):
    logger = logging.getLogger(__name__)
    world_size = int(get_world_size())
    if is_distributed():
        num_batches = int(math.ceil(float(num_batches) / float(world_size)))
        if int(max_steps) > 0:
            max_steps = int(math.ceil(float(max_steps) / float(world_size)))

    crr_cfg = cfg.get("crr") or {}
    if not isinstance(crr_cfg, dict):
        raise ValueError("crr must be a mapping (dict)")
    weight_type = str(crr_cfg.get("weight_type", "exp")).strip().lower()
    if weight_type not in {"exp", "binary"}:
        raise ValueError("crr.weight_type must be one of: exp | binary")
    advantage_baseline = str(crr_cfg.get("advantage_baseline", "max")).strip().lower()
    if advantage_baseline not in {"none", "max", "mean"}:
        raise ValueError("crr.advantage_baseline must be one of: none | max | mean")
    temperature = float(crr_cfg.get("temperature", 1.0))
    if weight_type == "exp" and (not (temperature > 0.0)):
        raise ValueError("crr.temperature must be > 0 when crr.weight_type=exp")
    tau = float(crr_cfg.get("tau", 0.005))
    gamma = float(crr_cfg.get("gamma", cfg.get("discount", 0.5)))
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("crr.gamma must be in [0, 1]")
    critic_loss_weight = float(crr_cfg.get("critic_loss_weight", 1.0))
    actor_lr = float(crr_cfg.get("actor_lr", cfg.get("lr", 0.005)))
    critic_lr = float(crr_cfg.get("critic_lr", cfg.get("lr", 0.005)))

    pointwise_critic_use, pointwise_critic_arch, pointwise_mlp_cfg = validate_pointwise_critic_cfg(cfg)
    if advantage_baseline == "mean" and bool(pointwise_critic_use) and str(pointwise_critic_arch) != "dot":
        raise ValueError("crr.advantage_baseline=mean requires pointwise_critic.arch=dot when pointwise_critic.use=true")

    online = SASRecQNetworkRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        pointwise_critic_use=bool(pointwise_critic_use),
        pointwise_critic_arch=str(pointwise_critic_arch),
        pointwise_critic_mlp=pointwise_mlp_cfg,
    ).to(device)
    target = SASRecQNetworkRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        pointwise_critic_use=bool(pointwise_critic_use),
        pointwise_critic_arch=str(pointwise_critic_arch),
        pointwise_critic_mlp=pointwise_mlp_cfg,
    ).to(device)

    if is_distributed():
        local_rank = int(get_local_rank())
        online = DDP(online, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
        return m.module if hasattr(m, "module") else m

    target.load_state_dict(_unwrap(online).state_dict())
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    base = _unwrap(online)
    actor_opt = torch.optim.Adam(list(base.backbone_parameters()), lr=float(actor_lr))
    critic_opt = torch.optim.Adam(list(base.critic_parameters()), lr=float(critic_lr))

    total_step = 0
    early_patience = int(cfg.get("early_stopping_ep", 5))
    best_metric = float("-inf")
    epochs_since_improve = 0
    stop_training = False

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
            _ = time.perf_counter() - t0
        else:
            dl = []

        online.train()
        base = _unwrap(online)
        tgt = target
        pad_id = int(getattr(base, "pad_id", 0))
        vocab = int(item_num) + 1
        use_pointwise = bool(getattr(base, "pointwise_critic_use", False))
        use_mlp_critic = False
        mlp = None
        if use_pointwise:
            use_mlp_critic = str(getattr(base, "pointwise_critic_arch", "dot")) == "mlp"
            mlp = getattr(base, "pointwise_critic_mlp", None)
            if use_mlp_critic and mlp is None:
                raise RuntimeError("pointwise_critic_mlp is not initialized")

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
            is_buy = step["is_buy"].to(device, non_blocking=pin_memory).to(torch.long)
            valid_mask = step["valid_mask"].to(device, non_blocking=pin_memory)
            done_mask = step["done_mask"].to(device, non_blocking=pin_memory)

            step_count = int(valid_mask.sum().item())
            if step_count <= 0:
                continue

            action_flat = actions[valid_mask]
            is_buy_flat = is_buy[valid_mask]
            done_flat = done_mask[valid_mask].to(torch.float32)

            seqs = base.encode_seq(states_x)
            with torch.no_grad():
                seqs_tgt = tgt.encode_seq(states_x)
            seqs_next = torch.zeros_like(seqs)
            seqs_next[:, :-1, :] = seqs[:, 1:, :]
            seqs_curr_flat = seqs[valid_mask]
            seqs_next_flat = seqs_next[valid_mask]
            seqs_next_tgt = torch.zeros_like(seqs_tgt)
            seqs_next_tgt[:, :-1, :] = seqs_tgt[:, 1:, :]
            seqs_next_tgt_flat = seqs_next_tgt[valid_mask]

            logits_curr = seqs_curr_flat @ base.item_emb.weight.t()
            logits_next = seqs_next_flat @ base.item_emb.weight.t()
            if pad_id >= 0 and pad_id < vocab:
                logits_curr[:, pad_id] = float("-inf")
                logits_next[:, pad_id] = float("-inf")

            if reward_fn == "ndcg":
                with torch.no_grad():
                    reward_flat = ndcg_reward_from_logits(logits_curr.detach(), action_flat)
            else:
                reward_flat = torch.where(is_buy_flat == 1, float(reward_buy), float(reward_click)).to(torch.float32)

            a_star = logits_next.argmax(dim=1)
            if use_pointwise:
                with torch.no_grad():
                    q_tp1 = tgt.q_value(seqs_next_tgt_flat, a_star)
                    y = reward_flat + float(gamma) * (1.0 - done_flat) * q_tp1

                seqs_curr_det = seqs_curr_flat.detach()
                if use_mlp_critic:
                    emb_det = base.item_emb(action_flat).detach()
                    q_sa = mlp(torch.cat([seqs_curr_det, emb_det], dim=-1)).squeeze(-1)
                else:
                    emb_det = base.item_emb(action_flat).detach()
                    q_sa = (seqs_curr_det * emb_det).sum(dim=-1) + base.head_q.bias[action_flat]
                critic_loss = F.mse_loss(q_sa, y, reduction="mean")
            else:
                seqs_curr_det = seqs_curr_flat.detach()
                q_curr_full = base.head_q(seqs_curr_det)
                if pad_id >= 0 and pad_id < vocab:
                    q_curr_full[:, pad_id] = float("-inf")
                q_sa = q_curr_full.gather(1, action_flat[:, None]).squeeze(1)
                with torch.no_grad():
                    q_next_tgt_full = tgt.head_q(seqs_next_tgt_flat)
                    if pad_id >= 0 and pad_id < vocab:
                        q_next_tgt_full[:, pad_id] = float("-inf")
                    q_tp1 = q_next_tgt_full.gather(1, a_star[:, None]).squeeze(1)
                    y = reward_flat + float(gamma) * (1.0 - done_flat) * q_tp1
                critic_loss = F.mse_loss(q_sa, y, reduction="mean")

            with torch.no_grad():
                q_sa_det = q_sa.detach()
                if advantage_baseline == "none":
                    baseline = torch.zeros_like(q_sa_det)
                elif advantage_baseline == "max":
                    a_greedy = logits_curr.argmax(dim=1)
                    if use_pointwise:
                        baseline = base.q_value(seqs_curr_flat, a_greedy).detach()
                    else:
                        baseline = q_curr_full.gather(1, a_greedy[:, None]).squeeze(1).detach()
                else:
                    pi = F.softmax(logits_curr, dim=1)
                    if use_pointwise:
                        q_full = logits_curr + base.head_q.bias[None, :]
                        if pad_id >= 0 and pad_id < vocab:
                            q_full[:, pad_id] = 0.0
                        baseline = (pi * q_full).sum(dim=1).detach()
                    else:
                        q_full0 = q_curr_full.clone()
                        if pad_id >= 0 and pad_id < vocab:
                            q_full0[:, pad_id] = 0.0
                        baseline = (pi * q_full0).sum(dim=1).detach()
                advantage = q_sa_det - baseline

                if weight_type == "binary":
                    weight = (advantage > 0.0).to(dtype=torch.float32)
                else:
                    z = (advantage / float(temperature)).clamp(max=20.0)
                    weight = torch.exp(z)

            actor_loss = F.cross_entropy(logits_curr, action_flat, reduction="none")
            actor_loss = (actor_loss * weight).mean()

            critic_obj = float(critic_loss_weight) * critic_loss
            if bool(cfg.get("debug", False)) and (not torch.isfinite(critic_obj).all()):
                raise FloatingPointError(f"Non-finite critic loss (crr) at total_step={int(total_step)}")
            critic_opt.zero_grad(set_to_none=True)
            critic_obj.backward()
            critic_opt.step()

            if bool(cfg.get("debug", False)) and (not torch.isfinite(actor_loss).all()):
                raise FloatingPointError(f"Non-finite actor loss (crr) at total_step={int(total_step)}")
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()

            _soft_update_(target=tgt, online=base, tau=float(tau))

            total_step += int(step_count)

        eval_fn = evaluate if evaluate_fn is None else evaluate_fn
        val_metrics = eval_fn(
            online,
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
        if trial is not None:
            trial.report(float(metric), step=int(epoch_idx))
            if bool(getattr(trial, "should_prune", lambda: False)()):
                raise RuntimeError("optuna_pruned")

        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            if is_rank0():
                torch.save(_unwrap(online).state_dict(), run_dir / "best_model.pt")
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
        torch.save(_unwrap(online).state_dict(), best_path)
    return best_path


__all__ = ["train_crr"]
