import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from SASRecModules_torch import SASRecQNetworkTorch
import yaml

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train SA2C (Torch) with SASRec Q-network.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--early_stopping_ep", type=int, default=None, help="Patience epochs for early stopping.")
    parser.add_argument("--early_stopping_metric", type=str, default=None, help="Early stopping metric (ndcg@10).")
    parser.add_argument("--max_steps", type=int, default=None, help="If set, stop after this many update steps.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging and NaN checks (overrides config).")
    return parser.parse_args()


def _default_config():
    return {
        "seed": 0,
        "epoch": 50,
        "data": "data",
        "batch_size_train": 256,
        "batch_size_val": 256,
        "num_workers_train": 0,
        "num_workers_val": 0,
        "device_id": 0,
        "hidden_factor": 64,
        "num_heads": 1,
        "num_blocks": 1,
        "dropout_rate": 0.1,
        "r_click": 0.2,
        "r_buy": 1.0,
        "r_negative": -0.0,
        "lr": 0.005,
        "lr_2": 0.001,
        "discount": 0.5,
        "neg": 10,
        "weight": 1.0,
        "smooth": 0.0,
        "clip": 0.0,
        "max_steps": 0,
        "debug": False,
        "early_stopping_ep": 5,
        "early_stopping_metric": "ndcg@10",
    }


def _load_config(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    cfg = _default_config()
    cfg.update(data)
    return cfg


def _apply_cli_overrides(cfg: dict, args):
    if args.early_stopping_ep is not None:
        cfg["early_stopping_ep"] = int(args.early_stopping_ep)
    if args.early_stopping_metric is not None:
        cfg["early_stopping_metric"] = str(args.early_stopping_metric)
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if bool(args.debug):
        cfg["debug"] = True
    return cfg


def _make_run_dir(config_path: str):
    config_name = Path(config_path).stem
    kaggle_dir = Path(__file__).resolve().parent
    run_dir = kaggle_dir / "logs" / "SA2C_SASRec_torch" / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return config_name, run_dir


def _configure_logging(run_dir: Path, debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(levelname)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
        force=True,
    )


def _dump_config(cfg: dict, run_dir: Path):
    with open(run_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        return itemlist + [pad_item] * (length - len(itemlist))
    return itemlist


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


def calculate_off(
    sorted_list,
    true_items,
    rewards,
    reward_click,
    off_click_ng,
    off_purchase_ng,
    off_prob_click,
    off_prob_purchase,
    pop_dict,
    topk=10,
):
    true_items = np.asarray(true_items)
    rewards = np.asarray(rewards)
    rec_list = sorted_list[:, -topk:]
    for j in range(len(true_items)):
        prob = float(pop_dict[true_items[j]])
        if rewards[j] == reward_click:
            off_prob_click[0] += 1.0 / prob
        else:
            off_prob_purchase[0] += 1.0 / prob
        if true_items[j] in rec_list[j]:
            rank = topk - int(np.argwhere(rec_list[j] == true_items[j])[0][0])
            if rewards[j] == reward_click:
                off_click_ng[0] += (1.0 / np.log2(rank + 1)) / prob
            else:
                off_purchase_ng[0] += (1.0 / np.log2(rank + 1)) / prob


def _sample_negative_actions(item_num, actions, neg, device):
    bsz = actions.shape[0]
    neg_actions = torch.randint(0, item_num, size=(bsz, neg), device=device)
    bad = neg_actions.eq(actions[:, None])
    while bad.any():
        neg_actions[bad] = torch.randint(0, item_num, size=(int(bad.sum().item()),), device=device)
        bad = neg_actions.eq(actions[:, None])
    return neg_actions


@torch.no_grad()
def evaluate(model, val_loader, reward_click, pop_dict, device, debug=False):
    total_clicks = 0.0
    total_purchase = 0.0
    topk = [5, 10, 15, 20]

    total_reward = [0.0, 0.0, 0.0, 0.0]
    hit_clicks = [0.0, 0.0, 0.0, 0.0]
    ndcg_clicks = [0.0, 0.0, 0.0, 0.0]
    hit_purchase = [0.0, 0.0, 0.0, 0.0]
    ndcg_purchase = [0.0, 0.0, 0.0, 0.0]

    off_prob_click = [0.0]
    off_prob_purchase = [0.0]
    off_click_ng = [0.0]
    off_purchase_ng = [0.0]

    model.eval()
    for state, len_state, action, reward in tqdm(
        val_loader,
        desc="val",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        state_t = state.to(device, non_blocking=True)
        len_state_t = len_state.to(device, non_blocking=True)
        _, ce_logits = model(state_t, len_state_t)
        if debug and (not torch.isfinite(ce_logits).all()):
            raise FloatingPointError("Non-finite ce_logits during evaluation")
        kmax = int(max(topk))
        vals, idx = torch.topk(ce_logits, k=kmax, dim=1, largest=True, sorted=False)
        order = vals.argsort(dim=1)
        idx_sorted = idx.gather(1, order)
        sorted_list = idx_sorted.detach().cpu().numpy()

        actions = action.detach().cpu().numpy()
        rewards = reward.detach().cpu().numpy()
        total_clicks += float((rewards == reward_click).sum())
        total_purchase += float((rewards != reward_click).sum())
        calculate_hit(
            sorted_list,
            topk,
            actions,
            rewards,
            reward_click,
            total_reward,
            hit_clicks,
            ndcg_clicks,
            hit_purchase,
            ndcg_purchase,
        )
        calculate_off(
            sorted_list,
            actions,
            rewards,
            reward_click,
            off_click_ng,
            off_purchase_ng,
            off_prob_click,
            off_prob_purchase,
            pop_dict,
        )

    off_click_ng_val = off_click_ng[0] / off_prob_click[0] if off_prob_click[0] > 0 else 0.0
    off_purchase_ng_val = off_purchase_ng[0] / off_prob_purchase[0] if off_prob_purchase[0] > 0 else 0.0

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
    logger.info("#############################################################")
    logger.info("total clicks: %d, total purchase: %d", int(total_clicks), int(total_purchase))
    for k in topk:
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("clicks hr ndcg @ %d: %f, %f", k, float(click[f"hr@{k}"]), float(click[f"ndcg@{k}"]))
        logger.info("purchase hr ndcg @ %d: %f, %f", k, float(purchase[f"hr@{k}"]), float(purchase[f"ndcg@{k}"]))
    logger.info(
        "off-line corrected evaluation (click_ng,purchase_ng)@10: %f, %f",
        float(off_click_ng_val),
        float(off_purchase_ng_val),
    )
    logger.info("#############################################################")
    logger.info("")

    return {
        "topk": topk,
        "click": click,
        "purchase": purchase,
        "overall": overall,
        "off": {"click_ng@10": float(off_click_ng_val), "purchase_ng@10": float(off_purchase_ng_val)},
    }


class _ReplayBufferDataset(Dataset):
    def __init__(self, state, len_state, next_state, len_next_state, action, is_buy, is_done):
        super().__init__()
        self.state = state
        self.len_state = len_state
        self.next_state = next_state
        self.len_next_state = len_next_state
        self.action = action
        self.is_buy = is_buy
        self.is_done = is_done

    def __len__(self):
        return int(self.action.shape[0])

    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.len_state[idx],
            self.next_state[idx],
            self.len_next_state[idx],
            self.action[idx],
            self.is_buy[idx],
            self.is_done[idx],
        )


class _ValDataset(Dataset):
    def __init__(self, state, len_state, action, reward):
        super().__init__()
        self.state = state
        self.len_state = len_state
        self.action = action
        self.reward = reward

    def __len__(self):
        return int(self.action.shape[0])

    def __getitem__(self, idx):
        return self.state[idx], self.len_state[idx], self.action[idx], self.reward[idx]


def _build_eval_tensors(data_directory, split_df_name, state_size, item_num, reward_click, reward_buy):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, split_df_name))
    groups = eval_sessions.groupby("session_id", sort=False)

    states = []
    len_states = []
    actions = []
    rewards = []
    for _, group in groups:
        history = []
        for _, row in group.iterrows():
            state = list(history)
            len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
            states.append(pad_history(state, state_size, item_num))
            action = int(row["item_id"])
            is_buy = int(row["is_buy"])
            reward = reward_buy if is_buy == 1 else reward_click
            actions.append(action)
            rewards.append(float(reward))
            history.append(action)

    state_t = torch.from_numpy(np.asarray(states, dtype=np.int64))
    len_state_t = torch.from_numpy(np.asarray(len_states, dtype=np.int64))
    action_t = torch.from_numpy(np.asarray(actions, dtype=np.int64))
    reward_t = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
    return state_t, len_state_t, action_t, reward_t


def _build_eval_dataset(
    data_directory,
    split_df_name,
    state_size,
    item_num,
    reward_click,
    reward_buy,
):
    state, len_state, action, reward = _build_eval_tensors(
        data_directory=data_directory,
        split_df_name=split_df_name,
        state_size=state_size,
        item_num=item_num,
        reward_click=reward_click,
        reward_buy=reward_buy,
    )
    return _ValDataset(state, len_state, action, reward)


def _make_eval_loader(
    ds,
    batch_size,
    num_workers,
    pin_memory,
):
    persistent_workers = num_workers > 0
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )


def _metrics_row(metrics: dict, kind: str):
    topk = metrics["topk"]
    src = metrics[kind]
    row = {}
    for k in topk:
        row[f"hr@{k}"] = float(src.get(f"hr@{k}", 0.0))
        row[f"ndcg@{k}"] = float(src.get(f"ndcg@{k}", 0.0))
    return row


def main():
    args = parse_args()
    config_path = args.config
    cfg = _load_config(config_path)
    cfg = _apply_cli_overrides(cfg, args)
    if str(cfg.get("early_stopping_metric", "ndcg@10")) != "ndcg@10":
        raise ValueError("Only early_stopping_metric='ndcg@10' is supported.")
    config_name, run_dir = _make_run_dir(config_path)
    _configure_logging(run_dir, debug=bool(cfg.get("debug", False)))
    _dump_config(cfg, run_dir)

    logger = logging.getLogger(__name__)
    logger.info("run_dir: %s", str(run_dir))

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_epochs = int(cfg.get("epoch", 50))
    max_steps = int(cfg.get("max_steps", 0))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg.get('device_id', 0))}")
    else:
        device = torch.device("cpu")

    data_directory = str(cfg.get("data", "data"))
    data_statis = pd.read_pickle(os.path.join(data_directory, "data_statis.df"))
    state_size = int(data_statis["state_size"][0])
    item_num = int(data_statis["item_num"][0])
    reward_click = float(cfg.get("r_click", 0.2))
    reward_buy = float(cfg.get("r_buy", 1.0))
    reward_negative = float(cfg.get("r_negative", -0.0))

    train_batch_size = int(cfg.get("batch_size_train", 256))
    val_batch_size = int(cfg.get("batch_size_val", 256))
    train_num_workers = int(cfg.get("num_workers_train", 0))
    val_num_workers = int(cfg.get("num_workers_val", 0))

    replay_buffer = pd.read_pickle(os.path.join(data_directory, "replay_buffer.df"))
    with open(os.path.join(data_directory, "pop_dict.txt"), "r") as f:
        pop_dict = eval(f.read())

    qn1 = SASRecQNetworkTorch(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    qn2 = SASRecQNetworkTorch(
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
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / train_batch_size)
    if num_batches <= 0:
        logger.warning(
            "num_batches=%d (num_rows=%d, train_batch_size=%d) -> no training batches will run; metrics will be static",
            int(num_batches),
            int(num_rows),
            int(train_batch_size),
        )

    t0 = time.perf_counter()
    state_all = torch.from_numpy(np.asarray(replay_buffer["state"].tolist(), dtype=np.int64))
    len_state_all = torch.from_numpy(np.asarray(replay_buffer["len_state"].to_numpy(), dtype=np.int64))
    next_state_all = torch.from_numpy(np.asarray(replay_buffer["next_state"].tolist(), dtype=np.int64))
    len_next_state_all = torch.from_numpy(np.asarray(replay_buffer["len_next_states"].to_numpy(), dtype=np.int64))
    action_all = torch.from_numpy(np.asarray(replay_buffer["action"].to_numpy(), dtype=np.int64))
    is_buy_all = torch.from_numpy(np.asarray(replay_buffer["is_buy"].to_numpy(), dtype=np.int64))
    done_all = torch.from_numpy(np.asarray(replay_buffer["is_done"].to_numpy(), dtype=np.bool_))

    ds = _ReplayBufferDataset(
        state=state_all,
        len_state=len_state_all,
        next_state=next_state_all,
        len_next_state=len_next_state_all,
        action=action_all,
        is_buy=is_buy_all,
        is_done=done_all,
    )
    train_ds_s = time.perf_counter() - t0

    pin_memory = device.type == "cuda"
    train_persistent_workers = train_num_workers > 0
    t0 = time.perf_counter()
    val_ds = _build_eval_dataset(
        data_directory=data_directory,
        split_df_name="sampled_val.df",
        state_size=state_size,
        item_num=item_num,
        reward_click=reward_click,
        reward_buy=reward_buy,
    )
    val_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    val_dl = _make_eval_loader(val_ds, val_batch_size, val_num_workers, pin_memory)
    val_dl_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_ds = _build_eval_dataset(
        data_directory=data_directory,
        split_df_name="sampled_test.df",
        state_size=state_size,
        item_num=item_num,
        reward_click=reward_click,
        reward_buy=reward_buy,
    )
    test_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_dl = _make_eval_loader(test_ds, val_batch_size, val_num_workers, pin_memory)
    test_dl_s = time.perf_counter() - t0

    behavior_prob_table = torch.full((item_num + 1,), 1.0, dtype=torch.float32)
    for k, v in pop_dict.items():
        kk = int(k)
        if 0 <= kk <= item_num:
            behavior_prob_table[kk] = float(v)
    behavior_prob_table = behavior_prob_table.to(device)

    early_patience = int(cfg.get("early_stopping_ep", 5))
    best_metric = float("-inf")
    epochs_since_improve = 0
    stop_training = False

    for epoch_idx in range(num_epochs):
        if bool(cfg.get("debug", False)):
            logger.debug(
                "epoch=%d/%d num_rows=%d train_batch_size=%d num_batches=%d total_step=%d",
                int(epoch_idx + 1),
                int(num_epochs),
                int(num_rows),
                int(train_batch_size),
                int(num_batches),
                int(total_step),
            )
        sampler = RandomSampler(ds, replacement=True, num_samples=num_batches * int(train_batch_size))
        t0 = time.perf_counter()
        dl = DataLoader(
            ds,
            batch_size=int(train_batch_size),
            sampler=sampler,
            num_workers=train_num_workers,
            pin_memory=pin_memory,
            persistent_workers=train_persistent_workers,
            drop_last=True,
        )
        train_dl_s = time.perf_counter() - t0
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
            state, len_state, next_state, len_next_state, action, is_buy, is_done = batch
            reward = torch.where(is_buy == 1, reward_buy, reward_click).to(torch.float32)
            discount = torch.full_like(reward, float(cfg.get("discount", 0.5)), dtype=torch.float32)

            pointer = np.random.randint(0, 2)
            if pointer == 0:
                main_qn, target_qn = qn1, qn2
                opt1, opt2 = opt1_qn1, opt2_qn1
            else:
                main_qn, target_qn = qn2, qn1
                opt1, opt2 = opt1_qn2, opt2_qn2

            main_qn.train()
            target_qn.train()

            state_t = state.to(device, non_blocking=pin_memory)
            len_state_t = len_state.to(device, non_blocking=pin_memory)
            next_state_t = next_state.to(device, non_blocking=pin_memory)
            len_next_state_t = len_next_state.to(device, non_blocking=pin_memory)
            action_t = action.to(device, non_blocking=pin_memory).to(torch.long)
            reward_t = reward.to(device, non_blocking=pin_memory)
            discount_t = discount.to(device, non_blocking=pin_memory)
            done_t = is_done.to(device, non_blocking=pin_memory).to(torch.float32)

            neg_actions_t = _sample_negative_actions(item_num, action_t, int(cfg.get("neg", 10)), device=device)

            with torch.no_grad():
                q_next_target, _ = target_qn(next_state_t, len_next_state_t)
                q_next_selector, _ = main_qn(next_state_t, len_next_state_t)
                q_curr_target, _ = target_qn(state_t, len_state_t)

            q_values, ce_logits = main_qn(state_t, len_state_t)
            if bool(cfg.get("debug", False)):
                if not torch.isfinite(q_values).all():
                    raise FloatingPointError(f"Non-finite q_values at total_step={int(total_step)}")
                if not torch.isfinite(ce_logits).all():
                    raise FloatingPointError(f"Non-finite ce_logits at total_step={int(total_step)}")

            a_star = q_next_selector.argmax(dim=1)
            q_tp1 = q_next_target.gather(1, a_star[:, None]).squeeze(1)
            target_pos = reward_t + discount_t * q_tp1 * (1.0 - done_t)
            q_sa = q_values.gather(1, action_t[:, None]).squeeze(1)
            qloss_pos = ((q_sa - target_pos.detach()) ** 2).mean()

            a_star_curr = q_values.detach().argmax(dim=1)
            q_t_star = q_curr_target.gather(1, a_star_curr[:, None]).squeeze(1)
            target_neg = reward_negative + discount_t * q_t_star
            q_sneg = q_values.gather(1, neg_actions_t)
            qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

            ce_loss_pre = F.cross_entropy(ce_logits, action_t, reduction="none")

            if total_step < 15000:
                loss = qloss_pos + qloss_neg + ce_loss_pre.mean()
                if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                    raise FloatingPointError(f"Non-finite loss (phase1) at total_step={int(total_step)}")
                opt1.zero_grad(set_to_none=True)
                loss.backward()
                opt1.step()
                total_step += 1
            else:
                with torch.no_grad():
                    prob = F.softmax(ce_logits, dim=1).gather(1, action_t[:, None]).squeeze(1)
                behavior_prob = behavior_prob_table[action_t]
                ips = (prob / behavior_prob).clamp(0.1, 10.0).pow(float(cfg.get("smooth", 0.0)))

                with torch.no_grad():
                    q_pos_det = q_values.gather(1, action_t[:, None]).squeeze(1)
                    q_neg_det = q_values.gather(1, neg_actions_t).sum(dim=1)
                    q_avg = (q_pos_det + q_neg_det) / float(1 + int(cfg.get("neg", 10)))
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
                total_step += 1
        val_metrics = evaluate(qn1, val_dl, reward_click, pop_dict, device, debug=bool(cfg.get("debug", False)))
        metric = float(val_metrics["overall"].get("ndcg@10", 0.0))
        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            torch.save(qn1.state_dict(), run_dir / "best_model.pt")
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
        torch.save(qn1.state_dict(), best_path)

    best_model = SASRecQNetworkTorch(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    val_best = evaluate(best_model, val_dl, reward_click, pop_dict, device, debug=bool(cfg.get("debug", False)))
    test_best = evaluate(best_model, test_dl, reward_click, pop_dict, device, debug=bool(cfg.get("debug", False)))

    val_click = _metrics_row(val_best, "click")
    test_click = _metrics_row(test_best, "click")
    val_purchase = _metrics_row(val_best, "purchase")
    test_purchase = _metrics_row(test_best, "purchase")

    col_order = []
    for k in val_best["topk"]:
        col_order.extend([f"val/hr@{k}", f"test/hr@{k}", f"val/ndcg@{k}", f"test/ndcg@{k}"])

    click_row = {}
    purchase_row = {}
    for k in val_best["topk"]:
        click_row[f"val/hr@{k}"] = float(val_click.get(f"hr@{k}", 0.0))
        click_row[f"test/hr@{k}"] = float(test_click.get(f"hr@{k}", 0.0))
        click_row[f"val/ndcg@{k}"] = float(val_click.get(f"ndcg@{k}", 0.0))
        click_row[f"test/ndcg@{k}"] = float(test_click.get(f"ndcg@{k}", 0.0))

        purchase_row[f"val/hr@{k}"] = float(val_purchase.get(f"hr@{k}", 0.0))
        purchase_row[f"test/hr@{k}"] = float(test_purchase.get(f"hr@{k}", 0.0))
        purchase_row[f"val/ndcg@{k}"] = float(val_purchase.get(f"ndcg@{k}", 0.0))
        purchase_row[f"test/ndcg@{k}"] = float(test_purchase.get(f"ndcg@{k}", 0.0))

    df_clicks = pd.DataFrame([click_row], index=["metrics"]).loc[:, col_order]
    df_purchase = pd.DataFrame([purchase_row], index=["metrics"]).loc[:, col_order]
    df_clicks.to_csv(run_dir / "results_clicks.csv", index=False)
    df_purchase.to_csv(run_dir / "results_purchase.csv", index=False)


if __name__ == "__main__":
    main()


