from __future__ import annotations

import logging

import numpy as np
import torch

from .data_utils.sessions import make_shifted_batch_from_sessions
from .utils import tqdm


def get_metric_value(metrics: dict, key: str) -> float:
    k = str(key).strip()
    if k.startswith("ndcg@") or k.startswith("hr@"):
        k = f"overall.{k}"
    parts = [p for p in k.split(".") if p]
    if len(parts) != 2:
        raise ValueError(f"metric must look like 'overall.ndcg@10', got {key!r}")
    section, name = parts[0], parts[1]
    if section not in {"overall", "click", "purchase"}:
        raise ValueError(f"metric section must be overall|click|purchase, got {section!r}")
    return float(metrics.get(section, {}).get(name, 0.0))


def calculate_hit(
    sorted_list,
    topk,
    true_items,
    rewards,
    r_click,
    total_reward,
    hit_click,
    ndcg_click,
    hit_purchase,
    ndcg_purchase,
):
    true_items = np.asarray(true_items)
    rewards = np.asarray(rewards)
    for i, k in enumerate(topk):
        rec_list = sorted_list[:, -k:]
        hits = (rec_list == true_items[:, None]).any(axis=1)
        hit_idx = np.where(hits)[0]
        if hit_idx.size == 0:
            continue
        pos = rec_list[hit_idx] == true_items[hit_idx, None]
        rank = k - pos.argmax(axis=1)
        total_reward[i] += rewards[hit_idx].sum()
        is_click = rewards[hit_idx] == r_click
        is_purchase = ~is_click
        if is_click.any():
            hit_click[i] += float(is_click.sum())
            ndcg_click[i] += float((1.0 / np.log2(rank[is_click] + 1)).sum())
        if is_purchase.any():
            hit_purchase[i] += float(is_purchase.sum())
            ndcg_purchase[i] += float((1.0 / np.log2(rank[is_purchase] + 1)).sum())


def extract_ce_logits_seq(model_output):
    if isinstance(model_output, (tuple, list)):
        if len(model_output) != 2:
            raise ValueError("Expected model output (q_seq, ce_logits_seq) or ce_logits_seq")
        return model_output[1]
    return model_output


def ndcg_reward_from_logits(ce_logits: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
    if ce_logits.ndim != 2:
        raise ValueError(f"Expected ce_logits shape [B,V], got {tuple(ce_logits.shape)}")
    if action_t.ndim != 1:
        raise ValueError(f"Expected action_t shape [B], got {tuple(action_t.shape)}")
    bsz, vocab = ce_logits.shape
    if action_t.shape[0] != bsz:
        raise ValueError(f"Batch mismatch: ce_logits[0]={bsz} action_t[0]={int(action_t.shape[0])}")
    if not torch.is_floating_point(ce_logits):
        ce_logits = ce_logits.to(torch.float32)
    action_t = action_t.to(torch.long)
    if torch.any(action_t < 0) or torch.any(action_t >= vocab):
        bad = action_t[(action_t < 0) | (action_t >= vocab)]
        raise ValueError(f"action_t contains out-of-range ids (vocab={vocab}), e.g. {bad[:8].tolist()}")
    target = ce_logits.gather(1, action_t[:, None]).squeeze(1)
    rank = (ce_logits > target[:, None]).sum(dim=1).to(torch.float32) + 1.0
    return 1.0 / torch.log2(rank + 1.0)


@torch.no_grad()
def evaluate(
    model,
    session_loader,
    reward_click,
    reward_buy,
    device,
    *,
    split: str = "val",
    state_size: int,
    item_num: int,
    purchase_only: bool = False,
    debug: bool = False,
    epoch=None,
    num_epochs=None,
):
    total_clicks = 0.0
    total_purchase = 0.0
    topk = [5, 10, 15, 20]

    total_reward = [0.0, 0.0, 0.0, 0.0]
    hit_clicks = [0.0, 0.0, 0.0, 0.0]
    ndcg_clicks = [0.0, 0.0, 0.0, 0.0]
    hit_purchase = [0.0, 0.0, 0.0, 0.0]
    ndcg_purchase = [0.0, 0.0, 0.0, 0.0]

    model.eval()
    for items_pad, is_buy_pad, lengths in tqdm(
        session_loader,
        desc=str(split),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
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
        actions = step["actions"].to(device, non_blocking=True)
        is_buy = step["is_buy"].to(device, non_blocking=True)
        valid_mask = step["valid_mask"].to(device, non_blocking=True)

        ce_logits_seq = extract_ce_logits_seq(model(states_x))
        if debug and (not torch.isfinite(ce_logits_seq).all()):
            raise FloatingPointError("Non-finite ce_logits during evaluation")

        ce_logits = ce_logits_seq[valid_mask]
        action_t = actions[valid_mask]
        is_buy_t = is_buy[valid_mask]
        reward_t = torch.where(is_buy_t == 1, float(reward_buy), float(reward_click)).to(torch.float32)

        kmax = int(max(topk))
        vals, idx = torch.topk(ce_logits, k=kmax, dim=1, largest=True, sorted=False)
        order = vals.argsort(dim=1)
        idx_sorted = idx.gather(1, order)
        sorted_list = idx_sorted.detach().cpu().numpy()

        actions_np = action_t.detach().cpu().numpy()
        rewards_np = reward_t.detach().cpu().numpy()
        total_clicks += float((rewards_np == reward_click).sum())
        total_purchase += float((rewards_np != reward_click).sum())
        calculate_hit(
            sorted_list,
            topk,
            actions_np,
            rewards_np,
            reward_click,
            total_reward,
            hit_clicks,
            ndcg_clicks,
            hit_purchase,
            ndcg_purchase,
        )

    click = {}
    purchase = {}
    overall = {}
    denom_all = float(total_clicks + total_purchase)
    for i, k in enumerate(topk):
        hr_click = hit_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        hr_purchase = hit_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        ng_click = ndcg_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        ng_purchase = ndcg_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        click[f"hr@{k}"] = float(hr_click)
        click[f"ndcg@{k}"] = float(ng_click)
        purchase[f"hr@{k}"] = float(hr_purchase)
        purchase[f"ndcg@{k}"] = float(ng_purchase)
        overall[f"ndcg@{k}"] = float((ndcg_clicks[i] + ndcg_purchase[i]) / denom_all) if denom_all > 0 else 0.0

    logger = logging.getLogger(__name__)
    if epoch is not None and num_epochs is not None:
        prefix = f"epoch {int(epoch)}/{int(num_epochs)} "
    elif epoch is not None:
        prefix = f"epoch {int(epoch)} "
    else:
        prefix = ""
    logger.info("#############################################################")
    logger.info("%s%s metrics", prefix, str(split))
    logger.info("total clicks: %d, total purchase: %d", int(total_clicks), int(total_purchase))
    for k in topk:
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("clicks hr ndcg @ %d: %f, %f", k, float(click[f"hr@{k}"]), float(click[f"ndcg@{k}"]))
        logger.info("purchase hr ndcg @ %d: %f, %f", k, float(purchase[f"hr@{k}"]), float(purchase[f"ndcg@{k}"]))
    logger.info("#############################################################")
    logger.info("")

    return {
        "topk": topk,
        "click": click,
        "purchase": purchase,
        "overall": overall,
    }


@torch.no_grad()
def evaluate_loo(
    model,
    session_loader,
    reward_click,
    reward_buy,
    device,
    *,
    split: str = "val",
    state_size: int,
    item_num: int,
    purchase_only: bool = False,
    debug: bool = False,
    epoch=None,
    num_epochs=None,
):
    total_clicks = 0.0
    total_purchase = 0.0
    topk = [5, 10, 15, 20]

    total_reward = [0.0, 0.0, 0.0, 0.0]
    hit_clicks = [0.0, 0.0, 0.0, 0.0]
    ndcg_clicks = [0.0, 0.0, 0.0, 0.0]
    hit_purchase = [0.0, 0.0, 0.0, 0.0]
    ndcg_purchase = [0.0, 0.0, 0.0, 0.0]

    model.eval()
    for items_pad, is_buy_pad, lengths in tqdm(
        session_loader,
        desc=str(split),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
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
        actions = step["actions"].to(device, non_blocking=True)
        is_buy = step["is_buy"].to(device, non_blocking=True)
        done_mask = step["done_mask"].to(device, non_blocking=True)

        ce_logits_seq = extract_ce_logits_seq(model(states_x))
        if debug and (not torch.isfinite(ce_logits_seq).all()):
            raise FloatingPointError("Non-finite ce_logits during evaluation")

        ce_logits = ce_logits_seq[done_mask]
        action_t = actions[done_mask]
        is_buy_t = is_buy[done_mask]
        reward_t = torch.where(is_buy_t == 1, float(reward_buy), float(reward_click)).to(torch.float32)

        kmax = int(max(topk))
        vals, idx = torch.topk(ce_logits, k=kmax, dim=1, largest=True, sorted=False)
        order = vals.argsort(dim=1)
        idx_sorted = idx.gather(1, order)
        sorted_list = idx_sorted.detach().cpu().numpy()

        actions_np = action_t.detach().cpu().numpy()
        rewards_np = reward_t.detach().cpu().numpy()
        total_clicks += float((rewards_np == reward_click).sum())
        total_purchase += float((rewards_np != reward_click).sum())
        calculate_hit(
            sorted_list,
            topk,
            actions_np,
            rewards_np,
            reward_click,
            total_reward,
            hit_clicks,
            ndcg_clicks,
            hit_purchase,
            ndcg_purchase,
        )

    click = {}
    purchase = {}
    overall = {}
    denom_all = float(total_clicks + total_purchase)
    for i, k in enumerate(topk):
        hr_click = hit_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        hr_purchase = hit_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        ng_click = ndcg_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        ng_purchase = ndcg_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        click[f"hr@{k}"] = float(hr_click)
        click[f"ndcg@{k}"] = float(ng_click)
        purchase[f"hr@{k}"] = float(hr_purchase)
        purchase[f"ndcg@{k}"] = float(ng_purchase)
        overall[f"ndcg@{k}"] = float((ndcg_clicks[i] + ndcg_purchase[i]) / denom_all) if denom_all > 0 else 0.0

    logger = logging.getLogger(__name__)
    if epoch is not None and num_epochs is not None:
        prefix = f"epoch {int(epoch)}/{int(num_epochs)} "
    elif epoch is not None:
        prefix = f"epoch {int(epoch)} "
    else:
        prefix = ""
    logger.info("#############################################################")
    logger.info("%s%s metrics", prefix, str(split))
    logger.info("total clicks: %d, total purchase: %d", int(total_clicks), int(total_purchase))
    for k in topk:
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("clicks hr ndcg @ %d: %f, %f", k, float(click[f"hr@{k}"]), float(click[f"ndcg@{k}"]))
        logger.info("purchase hr ndcg @ %d: %f, %f", k, float(purchase[f"hr@{k}"]), float(purchase[f"ndcg@{k}"]))
    logger.info("#############################################################")
    logger.info("")

    return {
        "topk": topk,
        "click": click,
        "purchase": purchase,
        "overall": overall,
    }


def metrics_row(metrics: dict, kind: str):
    topk = metrics["topk"]
    src = metrics[kind]
    row = {}
    for k in topk:
        row[f"hr@{k}"] = float(src.get(f"hr@{k}", 0.0))
        row[f"ndcg@{k}"] = float(src.get(f"ndcg@{k}", 0.0))
    return row


def overall_row(metrics: dict):
    topk = metrics["topk"]
    src = metrics["overall"]
    row = {}
    for k in topk:
        row[f"ndcg@{k}"] = float(src.get(f"ndcg@{k}", 0.0))
    return row


def summary_at_k_text(val_metrics: dict, test_metrics: dict, k: int):
    def g(m: dict, section: str, key: str):
        return float(m.get(section, {}).get(key, 0.0))

    lines = [
        f"overall val/ndcg@{k}={g(val_metrics, 'overall', f'ndcg@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'overall', f'ndcg@{k}'):.6f}",
        f"click   val/hr@{k}={g(val_metrics, 'click', f'hr@{k}'):.6f} val/ndcg@{k}={g(val_metrics, 'click', f'ndcg@{k}'):.6f}  test/hr@{k}={g(test_metrics, 'click', f'hr@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'click', f'ndcg@{k}'):.6f}",
        f"purchase val/hr@{k}={g(val_metrics, 'purchase', f'hr@{k}'):.6f} val/ndcg@{k}={g(val_metrics, 'purchase', f'ndcg@{k}'):.6f}  test/hr@{k}={g(test_metrics, 'purchase', f'hr@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'purchase', f'ndcg@{k}'):.6f}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "evaluate",
    "evaluate_loo",
    "calculate_hit",
    "ndcg_reward_from_logits",
    "get_metric_value",
    "metrics_row",
    "overall_row",
    "summary_at_k_text",
]

