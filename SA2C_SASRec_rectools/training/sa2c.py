from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from ..data.sessions import make_session_loader, make_shifted_batch_from_sessions
from ..metrics import evaluate, ndcg_reward_from_logits
from ..models import SASRecQNetworkRectools
from ..utils import tqdm


def sample_negative_actions(min_id: int, max_id_exclusive: int, actions, neg, device):
    bsz = actions.shape[0]
    neg_actions = torch.randint(int(min_id), int(max_id_exclusive), size=(bsz, neg), device=device)
    bad = neg_actions.eq(actions[:, None])
    while bad.any():
        neg_actions[bad] = torch.randint(int(min_id), int(max_id_exclusive), size=(int(bad.sum().item()),), device=device)
        bad = neg_actions.eq(actions[:, None])
    return neg_actions


def train_sa2c(
    *,
    cfg: dict,
    train_ds,
    val_dl,
    pop_dict_path: Path,
    run_dir: Path,
    device: torch.device,
    reward_click: float,
    reward_buy: float,
    reward_negative: float,
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
) -> tuple[Path, Path | None]:
    logger = logging.getLogger(__name__)
    with open(str(pop_dict_path), "r") as f:
        pop_dict = eval(f.read())

    qn1 = SASRecQNetworkRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    qn2 = SASRecQNetworkRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)

    opt1_qn1 = torch.optim.Adam(qn1.parameters(), lr=float(cfg.get("lr", 0.005)))
    opt2_qn1 = torch.optim.Adam(qn1.parameters(), lr=float(cfg.get("lr_2", 0.001)))
    opt1_qn2 = torch.optim.Adam(qn2.parameters(), lr=float(cfg.get("lr", 0.005)))
    opt2_qn2 = torch.optim.Adam(qn2.parameters(), lr=float(cfg.get("lr_2", 0.001)))

    total_step = 0

    behavior_prob_table = torch.full((item_num + 1,), 1.0, dtype=torch.float32)
    for k, v in pop_dict.items():
        kk = int(k)
        if 0 <= kk < item_num:
            behavior_prob_table[kk + 1] = float(v)
    behavior_prob_table = behavior_prob_table.to(device)

    early_patience = int(cfg.get("early_stopping_ep", 5))
    warmup_patience_cfg = cfg.get("early_stopping_warmup_ep", None)
    warmup_patience = None if warmup_patience_cfg is None else int(warmup_patience_cfg)
    use_auto_warmup = warmup_patience is not None
    best_metric = float("-inf")
    stop_training = False

    best_metric_warmup = float("-inf")
    epochs_since_improve_warmup = 0
    best_warmup_path = run_dir / "best_warmup_model.pt"

    best_metric_phase2 = float("-inf")
    epochs_since_improve_phase2 = 0

    phase = "warmup" if use_auto_warmup else "scheduled"
    warmup_best_metric_scalar = float("-inf")
    warmup_baseline_finalized = False
    entered_finetune = False

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

        for batch_idx, batch in enumerate(
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
            discount = torch.full((step_count,), float(cfg.get("discount", 0.5)), dtype=torch.float32, device=device)
            warmup_epochs = float(cfg.get("warmup_epochs", 0.0))
            epoch_progress = float(epoch_idx) + (float(batch_idx) / float(max(1, num_batches)))
            if phase == "scheduled":
                in_warmup = epoch_progress < warmup_epochs
            else:
                in_warmup = phase == "warmup"
            if (not in_warmup) and (not entered_finetune):
                entered_finetune = True
                if not warmup_baseline_finalized:
                    if np.isfinite(best_metric) and best_metric > float("-inf"):
                        warmup_best_metric_scalar = float(best_metric)
                        warmup_baseline_finalized = True
                    if phase == "scheduled":
                        best_metric_phase2 = float("-inf")
                        epochs_since_improve_phase2 = 0

            sampled_cfg = cfg.get("sampled_loss") or {}
            use_sampled_loss = bool(sampled_cfg.get("use", False))
            ce_n_neg = int(sampled_cfg.get("ce_n_negatives", 256))
            critic_n_neg = int(sampled_cfg.get("critic_n_negatives", 256))

            pointer = np.random.randint(0, 2)
            if pointer == 0:
                main_qn, target_qn = qn1, qn2
                opt1, opt2 = opt1_qn1, opt2_qn1
            else:
                main_qn, target_qn = qn2, qn1
                opt1, opt2 = opt1_qn2, opt2_qn2

            main_qn.train()
            target_qn.train()

            action_flat = actions[valid_mask]
            is_buy_flat = is_buy[valid_mask]
            done_flat = done_mask[valid_mask].to(torch.float32)

            if use_sampled_loss:
                seqs_main = main_qn.encode_seq(states_x)
                with torch.no_grad():
                    seqs_tgt = target_qn.encode_seq(states_x)

                seqs_next_main = torch.zeros_like(seqs_main)
                seqs_next_tgt = torch.zeros_like(seqs_tgt)
                seqs_next_main[:, :-1, :] = seqs_main[:, 1:, :]
                seqs_next_tgt[:, :-1, :] = seqs_tgt[:, 1:, :]

                seqs_curr_flat = seqs_main[valid_mask]
                seqs_curr_tgt_flat = seqs_tgt[valid_mask]
                seqs_next_selector_flat = seqs_next_main[valid_mask]
                seqs_next_target_flat = seqs_next_tgt[valid_mask]

                if bool(cfg.get("debug", False)):
                    if not torch.isfinite(seqs_curr_flat).all():
                        raise FloatingPointError(f"Non-finite seq encodings at total_step={int(total_step)}")

                crit_negs = sample_negative_actions(1, item_num + 1, action_flat, critic_n_neg, device=device)
                crit_cands = torch.cat([action_flat[:, None], crit_negs], dim=1)
                q_curr_c = main_qn.score_q_candidates(seqs_curr_flat, crit_cands)
                q_curr_tgt_c = target_qn.score_q_candidates(seqs_curr_tgt_flat, crit_cands)
                q_next_selector_c = main_qn.score_q_candidates(seqs_next_selector_flat, crit_cands)
                q_next_target_c = target_qn.score_q_candidates(seqs_next_target_flat, crit_cands)

                ce_negs = sample_negative_actions(1, item_num + 1, action_flat, ce_n_neg, device=device)
                ce_cands = torch.cat([action_flat[:, None], ce_negs], dim=1)
                ce_logits_c = main_qn.score_ce_candidates(seqs_curr_flat, ce_cands)

                if bool(cfg.get("debug", False)):
                    if not torch.isfinite(q_curr_c).all():
                        raise FloatingPointError(f"Non-finite q_values(cand) at total_step={int(total_step)}")
                    if not torch.isfinite(ce_logits_c).all():
                        raise FloatingPointError(f"Non-finite ce_logits(cand) at total_step={int(total_step)}")

                if reward_fn == "ndcg":
                    with torch.no_grad():
                        ce_full_seq = seqs_main @ main_qn.item_emb.weight.t()
                        ce_full_seq[:, :, int(getattr(main_qn, "pad_id", 0))] = float("-inf")
                        ce_flat_full = ce_full_seq[valid_mask]
                        reward_flat = ndcg_reward_from_logits(ce_flat_full.detach(), action_flat)
                else:
                    reward_flat = torch.where(is_buy_flat == 1, float(reward_buy), float(reward_click)).to(torch.float32)

                a_star_idx = q_next_selector_c.argmax(dim=1)
                q_tp1 = q_next_target_c.gather(1, a_star_idx[:, None]).squeeze(1)
                target_pos = reward_flat + discount * q_tp1 * (1.0 - done_flat)
                q_sa = q_curr_c[:, 0]
                qloss_pos = ((q_sa - target_pos.detach()) ** 2).mean()

                a_star_curr_idx = q_curr_c.detach().argmax(dim=1)
                q_t_star = q_curr_tgt_c.gather(1, a_star_curr_idx[:, None]).squeeze(1)
                target_neg = float(reward_negative) + discount * q_t_star
                q_sneg = q_curr_c[:, 1:]
                qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

                ce_loss_pre = F.cross_entropy(
                    ce_logits_c,
                    torch.zeros((int(ce_logits_c.shape[0]),), dtype=torch.long, device=device),
                    reduction="none",
                )
                neg_count = int(critic_n_neg)
            else:
                q_main_seq, ce_main_seq = main_qn(states_x)
                if bool(cfg.get("debug", False)):
                    if not torch.isfinite(q_main_seq).all():
                        raise FloatingPointError(f"Non-finite q_values at total_step={int(total_step)}")
                    if not torch.isfinite(ce_main_seq).all():
                        raise FloatingPointError(f"Non-finite ce_logits at total_step={int(total_step)}")

                with torch.no_grad():
                    q_tgt_seq, _ = target_qn(states_x)

                q_next_selector = torch.zeros_like(q_main_seq)
                q_next_target = torch.zeros_like(q_tgt_seq)
                q_next_selector[:, :-1, :] = q_main_seq[:, 1:, :]
                q_next_target[:, :-1, :] = q_tgt_seq[:, 1:, :]

                q_curr_flat = q_main_seq[valid_mask]
                ce_flat = ce_main_seq[valid_mask]
                q_curr_tgt_flat = q_tgt_seq[valid_mask]
                q_next_selector_flat = q_next_selector[valid_mask]
                q_next_target_flat = q_next_target[valid_mask]

                if reward_fn == "ndcg":
                    with torch.no_grad():
                        reward_flat = ndcg_reward_from_logits(ce_flat.detach(), action_flat)
                else:
                    reward_flat = torch.where(is_buy_flat == 1, float(reward_buy), float(reward_click)).to(torch.float32)

                a_star = q_next_selector_flat.argmax(dim=1)
                q_tp1 = q_next_target_flat.gather(1, a_star[:, None]).squeeze(1)
                target_pos = reward_flat + discount * q_tp1 * (1.0 - done_flat)
                q_sa = q_curr_flat.gather(1, action_flat[:, None]).squeeze(1)
                qloss_pos = ((q_sa - target_pos.detach()) ** 2).mean()

                a_star_curr = q_curr_flat.detach().argmax(dim=1)
                q_t_star = q_curr_tgt_flat.gather(1, a_star_curr[:, None]).squeeze(1)
                target_neg = float(reward_negative) + discount * q_t_star
                neg_count = int(cfg.get("neg", 10))
                neg_actions = sample_negative_actions(1, item_num + 1, action_flat, neg_count, device=device)
                q_sneg = q_curr_flat.gather(1, neg_actions)
                qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

                ce_loss_pre = F.cross_entropy(ce_flat, action_flat, reduction="none")

            if in_warmup:
                loss = qloss_pos + qloss_neg + ce_loss_pre.mean()
                if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                    raise FloatingPointError(f"Non-finite loss (phase1) at total_step={int(total_step)}")
                opt1.zero_grad(set_to_none=True)
                loss.backward()
                opt1.step()
                total_step += int(step_count)
            else:
                with torch.no_grad():
                    if use_sampled_loss:
                        prob = F.softmax(ce_logits_c, dim=1)[:, 0]
                    else:
                        prob = F.softmax(ce_flat, dim=1).gather(1, action_flat[:, None]).squeeze(1)
                behavior_prob = behavior_prob_table[action_flat]
                ips = (prob / behavior_prob).clamp(0.1, 10.0).pow(float(cfg.get("smooth", 0.0)))

                with torch.no_grad():
                    if use_sampled_loss:
                        q_pos_det = q_curr_c[:, 0]
                        q_neg_det = q_curr_c[:, 1:].sum(dim=1)
                        q_avg = (q_pos_det + q_neg_det) / float(1 + int(neg_count))
                    else:
                        q_pos_det = q_curr_flat.gather(1, action_flat[:, None]).squeeze(1)
                        q_neg_det = q_curr_flat.gather(1, neg_actions).sum(dim=1)
                        q_avg = (q_pos_det + q_neg_det) / float(1 + int(neg_count))
                    advantage = q_pos_det - q_avg
                    if float(cfg.get("clip", 0.0)) > 0:
                        advantage = advantage.clamp(-float(cfg.get("clip", 0.0)), float(cfg.get("clip", 0.0)))

                ce_loss_post = ips * ce_loss_pre * advantage
                loss = float(cfg.get("weight", 1.0)) * (qloss_pos + qloss_neg) + ce_loss_post.mean()
                if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                    raise FloatingPointError(f"Non-finite loss (phase2) at total_step={int(total_step)}")
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
                total_step += int(step_count)

        eval_fn = evaluate if evaluate_fn is None else evaluate_fn
        val_metrics = eval_fn(
            qn1,
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
        metric_1 = float(val_metrics["overall"].get("ndcg@10", 0.0))
        _logger = logging.getLogger(__name__)
        _prev_disabled = bool(getattr(_logger, "disabled", False))
        _logger.disabled = True
        try:
            val_metrics_2 = eval_fn(
                qn2,
                val_dl,
                reward_click,
                reward_buy,
                device,
                debug=bool(cfg.get("debug", False)),
                split="val(qn2)",
                state_size=state_size,
                item_num=item_num,
                purchase_only=purchase_only,
                epoch=int(epoch_idx + 1),
                num_epochs=int(num_epochs),
            )
        finally:
            _logger.disabled = _prev_disabled
        metric_2 = float(val_metrics_2["overall"].get("ndcg@10", 0.0))
        if metric_2 > metric_1:
            metric = float(metric_2)
            best_state_for_epoch = qn2.state_dict()
        else:
            metric = float(metric_1)
            best_state_for_epoch = qn1.state_dict()
        if metric > best_metric:
            best_metric = metric
            torch.save(best_state_for_epoch, run_dir / "best_model.pt")
            logger.info("best_model.pt updated (val ndcg@10=%f)", float(best_metric))

        if use_auto_warmup and phase == "warmup":
            if metric > best_metric_warmup:
                best_metric_warmup = metric
                epochs_since_improve_warmup = 0
                torch.save(best_state_for_epoch, best_warmup_path)
                logger.info("best_warmup_model.pt updated (val ndcg@10=%f)", float(best_metric_warmup))
            else:
                epochs_since_improve_warmup += 1
                logger.info(
                    "warmup no improvement (val ndcg@10=%f best=%f) patience=%d/%d",
                    float(metric),
                    float(best_metric_warmup),
                    int(epochs_since_improve_warmup),
                    int(warmup_patience),
                )
            if int(warmup_patience) > 0 and epochs_since_improve_warmup >= int(warmup_patience):
                warmup_best_metric_scalar = float(best_metric_warmup)
                warmup_baseline_finalized = True
                entered_finetune = True
                if best_warmup_path.exists():
                    qn1.load_state_dict(torch.load(best_warmup_path, map_location=device))
                    qn2.load_state_dict(torch.load(best_warmup_path, map_location=device))
                phase = "finetune"
                best_metric_phase2 = float("-inf")
                epochs_since_improve_phase2 = 0
                logger.info("warmup early stopping triggered -> switching to phase2 finetune")

        elif use_auto_warmup and phase == "finetune":
            if not warmup_baseline_finalized:
                warmup_best_metric_scalar = float(best_metric)
                warmup_baseline_finalized = True
            if np.isfinite(warmup_best_metric_scalar) and warmup_best_metric_scalar > float("-inf"):
                logger.info(
                    "val ndcg@10=%f (delta_vs_warmup=%+.6f, warmup_best=%f)",
                    float(metric),
                    float(metric - warmup_best_metric_scalar),
                    float(warmup_best_metric_scalar),
                )
            if metric > best_metric_phase2:
                best_metric_phase2 = metric
                epochs_since_improve_phase2 = 0
            else:
                epochs_since_improve_phase2 += 1
                logger.info(
                    "finetune no improvement (val ndcg@10=%f best=%f) patience=%d/%d",
                    float(metric),
                    float(best_metric_phase2),
                    int(epochs_since_improve_phase2),
                    int(early_patience),
                )
                if early_patience > 0 and epochs_since_improve_phase2 >= early_patience:
                    logger.info("finetune early stopping triggered")
                    break
        else:
            if (not use_auto_warmup) and (phase == "scheduled") and (not entered_finetune) and float(
                cfg.get("warmup_epochs", 0.0)
            ) > 0.0:
                if metric > best_metric_warmup:
                    best_metric_warmup = metric
                    torch.save(best_state_for_epoch, best_warmup_path)
                    logger.info("best_warmup_model.pt updated (val ndcg@10=%f)", float(best_metric_warmup))

            if entered_finetune:
                if (not warmup_baseline_finalized) and np.isfinite(metric):
                    warmup_best_metric_scalar = float(metric)
                    warmup_baseline_finalized = True
                if np.isfinite(warmup_best_metric_scalar) and warmup_best_metric_scalar > float("-inf"):
                    logger.info(
                        "val ndcg@10=%f (delta_vs_warmup=%+.6f, warmup_best=%f)",
                        float(metric),
                        float(metric - warmup_best_metric_scalar),
                        float(warmup_best_metric_scalar),
                    )
                if (not use_auto_warmup) and phase == "scheduled":
                    if metric > best_metric_phase2:
                        best_metric_phase2 = metric
                        epochs_since_improve_phase2 = 0
                    else:
                        epochs_since_improve_phase2 += 1
                        logger.info(
                            "finetune no improvement (val ndcg@10=%f best=%f) patience=%d/%d",
                            float(metric),
                            float(best_metric_phase2),
                            int(epochs_since_improve_phase2),
                            int(early_patience),
                        )
                        if early_patience > 0 and epochs_since_improve_phase2 >= early_patience:
                            logger.info("finetune early stopping triggered")
                            break
            else:
                warmup_best_metric_scalar = float(max(warmup_best_metric_scalar, metric))

        if stop_training:
            logger.info("max_steps reached; stopping")
            break

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(qn1.state_dict(), best_path)
    warmup_path = best_warmup_path if best_warmup_path.exists() else None
    return best_path, warmup_path


__all__ = ["train_sa2c"]

