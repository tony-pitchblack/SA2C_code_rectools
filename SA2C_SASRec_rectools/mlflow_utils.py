from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from dotenv import dotenv_values

from .config import resolve_ce_sampling, validate_pointwise_critic_cfg
from .data_utils.sessions import make_shifted_batch_from_sessions
from .metrics import ndcg_reward_from_logits
from .training.sampling import sample_global_uniform_negatives, sample_uniform_negatives


def _strip_wrapping_quotes(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def setup_mlflow_tracking(*, repo_root: Path, timeout_s: float = 2.0) -> str:
    env_path = repo_root / ".env"
    if not env_path.exists():
        raise RuntimeError(f"Missing required file: {env_path}")
    values = dotenv_values(env_path)
    host = values.get("MLFLOW_HOST", "")
    port = values.get("MLFLOW_PORT", "")
    host = _strip_wrapping_quotes(host).strip()
    port = _strip_wrapping_quotes(port).strip()
    if not host or not port:
        raise RuntimeError("Missing required .env variables: MLFLOW_HOST and/or MLFLOW_PORT")
    uri = f"http://{host}:{port}"
    mlflow.set_tracking_uri(uri)
    return uri


def format_experiment_name(*, dataset_name: str, eval_scheme: str | None, limit_chunks_pct: float | None) -> str:
    suffix = str(eval_scheme) if eval_scheme is not None else "default"
    name = f"{dataset_name}-{suffix}"
    if limit_chunks_pct is not None:
        pct = float(limit_chunks_pct) * 100.0
        name = f"{name}-{pct:g}%"
    return name


def _norm_metric_name(name: str) -> str:
    return str(name).replace("@", "_")


def flatten_eval_metrics_for_mlflow(*, split: str, metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for section in ("click", "purchase", "overall"):
        src = metrics.get(section, {})
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            if k == "topk":
                continue
            try:
                out[f"{split}_{section}/{_norm_metric_name(str(k))}"] = float(v)
            except Exception:
                continue
    return out


def log_metrics_dict(metrics: dict[str, float], *, step: int | None = None) -> None:
    for k, v in metrics.items():
        if step is None:
            mlflow.log_metric(str(k), float(v))
        else:
            mlflow.log_metric(str(k), float(v), step=int(step))


def compute_baseline_ce_loss(
    *,
    model,
    session_loader,
    device: torch.device,
    state_size: int,
    item_num: int,
    purchase_only: bool,
    cfg: dict,
    ce_vocab_pct: float | None,
) -> float:
    model.eval()
    _, _, _, ce_n_neg_eff = resolve_ce_sampling(cfg=cfg, item_num=int(item_num))
    total_loss = 0.0
    total_tokens = 0

    for items_pad, is_buy_pad, lengths in session_loader:
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
        states_x = step["states_x"].to(device, non_blocking=True)
        actions = step["actions"].to(device, non_blocking=True).to(torch.long)
        valid_mask = step["valid_mask"].to(device, non_blocking=True)
        action_flat = actions[valid_mask]
        n = int(action_flat.numel())
        if n <= 0:
            continue

        if ce_n_neg_eff is None:
            ce_logits_seq = model(states_x)
            ce_logits = ce_logits_seq[valid_mask]
            loss_sum = F.cross_entropy(ce_logits, action_flat, reduction="sum")
        else:
            base = model.module if hasattr(model, "module") else model
            seqs = base.encode_seq(states_x)
            seqs_flat = seqs[valid_mask]
            if ce_vocab_pct is not None:
                neg_ids = sample_global_uniform_negatives(n_neg=int(ce_n_neg_eff), item_num=int(item_num), device=device)
                pos_emb = base.item_emb(action_flat)
                pos_logits = (seqs_flat * pos_emb).sum(dim=-1)
                neg_emb = base.item_emb(neg_ids)
                neg_logits = seqs_flat @ neg_emb.t()
                neg_logits = neg_logits.masked_fill(neg_ids[None, :].eq(action_flat[:, None]), float("-inf"))
                cand_logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
            else:
                negs = sample_uniform_negatives(actions=action_flat, n_neg=int(ce_n_neg_eff), item_num=int(item_num))
                cand_ids = torch.cat([action_flat[:, None], negs], dim=1)
                cand_logits = base.score_ce_candidates(seqs_flat, cand_ids)
            loss_sum = F.cross_entropy(
                cand_logits,
                torch.zeros((int(cand_logits.shape[0]),), dtype=torch.long, device=device),
                reduction="sum",
            )

        total_loss += float(loss_sum.item())
        total_tokens += int(n)

    return float(total_loss / float(max(1, total_tokens)))


def compute_albert4rec_ce_loss(
    *,
    model,
    session_loader,
    device: torch.device,
    state_size: int,
    item_num: int,
    n_negatives: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    mask_id = int(item_num) + 1

    for input_ids, _is_buy in session_loader:
        input_ids = input_ids.to(device, non_blocking=True).to(torch.long)
        if int(input_ids.shape[1]) != int(state_size):
            raise ValueError(f"Expected input_ids shape [B,{int(state_size)}], got {tuple(input_ids.shape)}")

        nonpad = input_ids.ne(0)
        lengths = nonpad.sum(dim=1).to(torch.long)
        keep = lengths.gt(0)
        if not bool(keep.any()):
            continue
        if not bool(keep.all()):
            input_ids = input_ids[keep]
            lengths = lengths[keep]

        bsz = int(input_ids.shape[0])
        last_idx = (lengths - 1).clamp(min=0).to(torch.long)
        rows = torch.arange(bsz, device=device)
        pos = input_ids[rows, last_idx].to(torch.long)
        masked = input_ids.clone()
        masked[rows, last_idx] = int(mask_id)

        h = model(masked)
        h_last = h[rows, last_idx]
        if int(pos.numel()) <= 0:
            continue

        neg = torch.randint(1, int(item_num) + 1, size=(int(pos.shape[0]), int(n_negatives)), device=device, dtype=torch.long)
        bad = neg.eq(pos[:, None])
        while bool(bad.any()):
            k = int(bad.sum().item())
            neg[bad] = torch.randint(1, int(item_num) + 1, size=(k,), device=device, dtype=torch.long)
            bad = neg.eq(pos[:, None])
        cand = torch.cat([pos[:, None], neg], dim=1)
        logits = model.score_candidates(h_last, cand)
        loss_sum = F.cross_entropy(
            logits,
            torch.zeros((int(pos.shape[0]),), dtype=torch.long, device=device),
            reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += int(pos.shape[0])

    return float(total_loss / float(max(1, total_tokens)))


@dataclass(frozen=True)
class Sa2cLosses:
    total: float
    actor: float
    critic: float


def _sample_corrected_policy_index_argmax(ce_next_logits_c: torch.Tensor, mu_c: torch.Tensor, mu_eps: float) -> torch.Tensor:
    corr = ce_next_logits_c - torch.log(mu_c.clamp_min(float(mu_eps)))
    return corr.argmax(dim=1, keepdim=True)


def compute_sa2c_losses(
    *,
    model,
    session_loader,
    device: torch.device,
    state_size: int,
    item_num: int,
    purchase_only: bool,
    cfg: dict,
    reward_click: float,
    reward_buy: float,
    reward_negative: float,
    reward_fn: str,
    pop_dict: dict,
    phase: str,
    ce_vocab_pct: float | None,
) -> Sa2cLosses:
    if str(phase) not in {"phase1", "phase2"}:
        raise ValueError("phase must be one of: phase1 | phase2")

    use_pointwise_critic, pointwise_arch, pointwise_mlp_cfg = validate_pointwise_critic_cfg(cfg)
    sampled_cfg = cfg.get("sampled_loss") or {}
    use_sampled_loss = bool(sampled_cfg.get("use", False))
    critic_n_neg = int(sampled_cfg.get("critic_n_negatives", 256))
    critic_use_pop_policy = bool((cfg.get("critic_sampling") or {}).get("use_pop_policy", False))
    critic_mu_eps = float((cfg.get("critic_sampling") or {}).get("mu_eps", 1e-12))

    _, _, _, ce_n_neg_eff = resolve_ce_sampling(cfg=cfg, item_num=int(item_num))

    behavior_prob_table = torch.full((int(item_num) + 1,), 1.0, dtype=torch.float32, device=device)
    for k, v in (pop_dict or {}).items():
        kk = int(k)
        if 0 <= kk < int(item_num):
            behavior_prob_table[kk + 1] = float(v)

    model.eval()
    target = copy.deepcopy(model).eval()

    total_loss_sum = 0.0
    actor_sum = 0.0
    critic_sum = 0.0
    total_tokens = 0

    weight = float(cfg.get("weight", 1.0))
    smooth = float(cfg.get("smooth", 0.0))
    clip = float(cfg.get("clip", 0.0))
    discount_val = float(cfg.get("discount", 0.5))
    neg_default = int(cfg.get("neg", 10))

    for items_pad, is_buy_pad, lengths in session_loader:
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
        states_x = step["states_x"].to(device, non_blocking=True)
        actions = step["actions"].to(device, non_blocking=True).to(torch.long)
        is_buy = step["is_buy"].to(device, non_blocking=True).to(torch.long)
        valid_mask = step["valid_mask"].to(device, non_blocking=True)
        done_mask = step["done_mask"].to(device, non_blocking=True)

        action_flat = actions[valid_mask]
        is_buy_flat = is_buy[valid_mask]
        done_flat = done_mask[valid_mask].to(torch.float32)
        step_count = int(action_flat.numel())
        if step_count <= 0:
            continue
        discount = torch.full((step_count,), float(discount_val), dtype=torch.float32, device=device)

        use_pointwise_branch = bool(use_pointwise_critic)
        if use_sampled_loss or use_pointwise_branch:
            critic_n_neg_eff = int(critic_n_neg) if use_sampled_loss else int(neg_default)
            if critic_use_pop_policy:
                w = behavior_prob_table.to(dtype=torch.float32)
                w = w.clone()
                if w.numel() > 0:
                    w[0] = 0.0
                w = w.clamp_min(0.0)
                neg_actions = torch.multinomial(w, num_samples=step_count * critic_n_neg_eff, replacement=True).view(
                    step_count, critic_n_neg_eff
                )
                bad = neg_actions.eq(action_flat[:, None])
                while bool(bad.any()):
                    k = int(bad.sum().item())
                    neg_actions[bad] = torch.multinomial(w, num_samples=k, replacement=True)
                    bad = neg_actions.eq(action_flat[:, None])
                crit_negs = neg_actions.to(action_flat.dtype)
            else:
                crit_negs = sample_uniform_negatives(actions=action_flat, n_neg=int(critic_n_neg_eff), item_num=int(item_num))
            crit_cands = torch.cat([action_flat[:, None], crit_negs], dim=1)

            ce_cands = None
            if ce_n_neg_eff is not None and (ce_vocab_pct is None):
                ce_cands = torch.cat(
                    [action_flat[:, None], sample_uniform_negatives(actions=action_flat, n_neg=int(ce_n_neg_eff), item_num=int(item_num))],
                    dim=1,
                )

            seqs_main, q_curr_c, q_next_selector_c, ce_logits_c, ce_next_logits_c = model(
                states_x,
                valid_mask=valid_mask,
                crit_cands=crit_cands,
                ce_cands=ce_cands,
                return_full_ce=(ce_n_neg_eff is None),
                ce_next_cands=(crit_cands if critic_use_pop_policy else None),
            )
            with torch.no_grad():
                _seqs_tgt, q_curr_tgt_c, q_next_target_c, _ce_tgt, _ce_next_tgt = target(
                    states_x,
                    valid_mask=valid_mask,
                    crit_cands=crit_cands,
                )

            if ce_n_neg_eff is not None and (ce_vocab_pct is not None):
                base = model.module if hasattr(model, "module") else model
                seqs_curr_flat = seqs_main[valid_mask]
                neg_ids = sample_global_uniform_negatives(n_neg=int(ce_n_neg_eff), item_num=int(item_num), device=device)
                pos_emb = base.item_emb(action_flat)
                pos_logits = (seqs_curr_flat * pos_emb).sum(dim=-1)
                neg_emb = base.item_emb(neg_ids)
                neg_logits = seqs_curr_flat @ neg_emb.t()
                neg_logits = neg_logits.masked_fill(neg_ids[None, :].eq(action_flat[:, None]), float("-inf"))
                ce_logits_c = torch.cat([pos_logits[:, None], neg_logits], dim=1)
            if ce_logits_c is None:
                raise RuntimeError("Expected ce_logits_c to be computed")

            if reward_fn == "ndcg":
                with torch.no_grad():
                    if ce_n_neg_eff is not None:
                        base = model.module if hasattr(model, "module") else model
                        ce_full_seq = seqs_main @ base.item_emb.weight.t()
                        ce_full_seq[:, :, int(getattr(base, "pad_id", 0))] = float("-inf")
                        reward_flat = ndcg_reward_from_logits(ce_full_seq[valid_mask].detach(), action_flat)
                    else:
                        reward_flat = ndcg_reward_from_logits(ce_logits_c.detach(), action_flat)
            else:
                reward_flat = torch.where(is_buy_flat == 1, float(reward_buy), float(reward_click)).to(torch.float32)

            if critic_use_pop_policy:
                if ce_next_logits_c is None:
                    raise RuntimeError("Expected ce_next_logits_c for pop-policy critic")
                mu_c = behavior_prob_table[crit_cands]
                a_star_idx = _sample_corrected_policy_index_argmax(ce_next_logits_c, mu_c, critic_mu_eps)
                q_tp1 = q_next_target_c.gather(1, a_star_idx).squeeze(1)
            else:
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

            if ce_n_neg_eff is not None:
                ce_loss_pre = F.cross_entropy(
                    ce_logits_c,
                    torch.zeros((int(ce_logits_c.shape[0]),), dtype=torch.long, device=device),
                    reduction="none",
                )
                with torch.no_grad():
                    prob = F.softmax(ce_logits_c, dim=1)[:, 0]
            else:
                ce_loss_pre = F.cross_entropy(ce_logits_c, action_flat, reduction="none")
                with torch.no_grad():
                    prob = F.softmax(ce_logits_c, dim=1).gather(1, action_flat[:, None]).squeeze(1)
            neg_count = int(critic_n_neg_eff)
        else:
            q_main_seq, ce_main_seq = model(states_x)
            with torch.no_grad():
                q_tgt_seq, _ce_tgt_seq = target(states_x)

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
            neg_count = int(neg_default)
            neg_actions = sample_uniform_negatives(actions=action_flat, n_neg=int(neg_count), item_num=int(item_num))
            q_sneg = q_curr_flat.gather(1, neg_actions)
            qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

            if ce_n_neg_eff is None:
                ce_loss_pre = F.cross_entropy(ce_flat, action_flat, reduction="none")
                with torch.no_grad():
                    prob = F.softmax(ce_flat, dim=1).gather(1, action_flat[:, None]).squeeze(1)
            else:
                base = model.module if hasattr(model, "module") else model
                seqs = base.encode_seq(states_x)
                seqs_flat = seqs[valid_mask]
                if ce_vocab_pct is not None:
                    neg_ids = sample_global_uniform_negatives(n_neg=int(ce_n_neg_eff), item_num=int(item_num), device=device)
                    pos_emb = base.item_emb(action_flat)
                    pos_logits = (seqs_flat * pos_emb).sum(dim=-1)
                    neg_emb = base.item_emb(neg_ids)
                    neg_logits = seqs_flat @ neg_emb.t()
                    neg_logits = neg_logits.masked_fill(neg_ids[None, :].eq(action_flat[:, None]), float("-inf"))
                    ce_logits_loss = torch.cat([pos_logits[:, None], neg_logits], dim=1)
                else:
                    ce_negs = sample_uniform_negatives(actions=action_flat, n_neg=int(ce_n_neg_eff), item_num=int(item_num))
                    ce_cands = torch.cat([action_flat[:, None], ce_negs], dim=1)
                    ce_logits_loss = base.score_ce_candidates(seqs_flat, ce_cands)
                ce_loss_pre = F.cross_entropy(
                    ce_logits_loss,
                    torch.zeros((int(ce_logits_loss.shape[0]),), dtype=torch.long, device=device),
                    reduction="none",
                )
                with torch.no_grad():
                    prob = F.softmax(ce_logits_loss, dim=1)[:, 0]

        critic_term = float((qloss_pos + qloss_neg).item())
        if str(phase) == "phase1":
            actor_term = float(ce_loss_pre.mean().item())
            total = float((qloss_pos + qloss_neg + ce_loss_pre.mean()).item())
        else:
            behavior_prob = behavior_prob_table[action_flat]
            ips = (prob / behavior_prob).clamp(0.1, 10.0).pow(float(smooth))
            with torch.no_grad():
                if use_sampled_loss or use_pointwise_critic:
                    q_pos_det = q_curr_c[:, 0]
                    q_neg_det = q_curr_c[:, 1:].sum(dim=1)
                    q_avg = (q_pos_det + q_neg_det) / float(1 + int(neg_count))
                else:
                    q_pos_det = q_curr_flat.gather(1, action_flat[:, None]).squeeze(1)
                    q_neg_det = q_curr_flat.gather(1, neg_actions).sum(dim=1)
                    q_avg = (q_pos_det + q_neg_det) / float(1 + int(neg_count))
                advantage = q_pos_det - q_avg
                if float(clip) > 0:
                    advantage = advantage.clamp(-float(clip), float(clip))
            ce_loss_post = ips * ce_loss_pre * advantage
            actor_term = float(ce_loss_post.mean().item())
            total = float((float(weight) * (qloss_pos + qloss_neg) + ce_loss_post.mean()).item())

        total_loss_sum += float(total) * float(step_count)
        actor_sum += float(actor_term) * float(step_count)
        critic_sum += float(critic_term) * float(step_count)
        total_tokens += int(step_count)

    denom = float(max(1, total_tokens))
    return Sa2cLosses(total=float(total_loss_sum / denom), actor=float(actor_sum / denom), critic=float(critic_sum / denom))

