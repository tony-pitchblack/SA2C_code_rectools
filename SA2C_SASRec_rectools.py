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

from SASRecModules_rectools import SASRecQNetworkRectools
import yaml

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train SA2C (Rectools) with per-position SASRec logits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--early_stopping_ep", type=int, default=None, help="Patience epochs for early stopping.")
    parser.add_argument("--early_stopping_metric", type=str, default=None, help="Early stopping metric (ndcg@10).")
    parser.add_argument("--max_steps", type=int, default=None, help="If set, stop after this many update steps.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging and NaN checks (overrides config).")
    parser.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Force CPU, set batch_size=8, run 1 epoch, and skip writing val/test result files (keeps logging).",
    )
    return parser.parse_args()


def _default_config():
    return {
        "seed": 0,
        "epoch": 50,
        "dataset": "retailrocket",
        "data": "data",
        "purchase_only": False,
        "reward_fn": "click_buy",
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


def _make_run_dir(dataset_name: str, config_name: str):
    repo_root = Path(__file__).resolve().parent
    run_dir = repo_root / "logs" / "SA2C_SASRec_rectools" / dataset_name / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_dataset_root(dataset: str):
    repo_root = Path(__file__).resolve().parent
    if dataset == "yoochoose":
        return repo_root / "RC15"
    if dataset == "retailrocket":
        return repo_root / "Kaggle"
    raise ValueError("dataset must be one of: yoochoose | retailrocket")


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


def _sample_negative_actions(min_id: int, max_id_exclusive: int, actions, neg, device):
    bsz = actions.shape[0]
    neg_actions = torch.randint(int(min_id), int(max_id_exclusive), size=(bsz, neg), device=device)
    bad = neg_actions.eq(actions[:, None])
    while bad.any():
        neg_actions[bad] = torch.randint(
            int(min_id), int(max_id_exclusive), size=(int(bad.sum().item()),), device=device
        )
        bad = neg_actions.eq(actions[:, None])
    return neg_actions


def _ndcg_reward_from_logits(ce_logits: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
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


class _SessionDataset(Dataset):
    def __init__(self, data_directory: str, split_df_name: str):
        super().__init__()
        df = pd.read_pickle(os.path.join(data_directory, split_df_name))
        groups = df.groupby("session_id", sort=False)
        items_list = []
        is_buy_list = []
        for _, group in groups:
            items = torch.from_numpy(group["item_id"].to_numpy(dtype=np.int64, copy=True))
            is_buy = torch.from_numpy(group["is_buy"].to_numpy(dtype=np.int64, copy=True))
            if items.numel() == 0:
                continue
            items_list.append(items)
            is_buy_list.append(is_buy)
        self.items_list = items_list
        self.is_buy_list = is_buy_list

    def __len__(self):
        return int(len(self.items_list))

    def __getitem__(self, idx: int):
        return self.items_list[idx], self.is_buy_list[idx]


def _collate_sessions(batch, pad_item: int):
    items_list, is_buy_list = zip(*batch)
    lengths = torch.as_tensor([int(x.numel()) for x in items_list], dtype=torch.long)
    lmax = int(lengths.max().item()) if lengths.numel() > 0 else 0
    bsz = int(len(items_list))
    items_pad = torch.full((bsz, lmax), int(pad_item), dtype=torch.long)
    is_buy_pad = torch.zeros((bsz, lmax), dtype=torch.long)
    for i, (items, is_buy) in enumerate(zip(items_list, is_buy_list)):
        n = int(items.numel())
        if n == 0:
            continue
        items_pad[i, :n] = items
        is_buy_pad[i, :n] = is_buy
    return items_pad, is_buy_pad, lengths


def _make_session_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    pad_item: int,
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
        collate_fn=lambda b: _collate_sessions(b, pad_item=int(pad_item)),
    )


def _make_shifted_batch_from_sessions(
    items_pad: torch.Tensor,
    is_buy_pad: torch.Tensor,
    lengths: torch.Tensor,
    *,
    state_size: int,
    old_pad_item: int,
    purchase_only: bool,
):
    bsz, lmax = items_pad.shape
    s = int(state_size)
    if lmax == 0:
        return None

    pos = torch.arange(s, device=items_pad.device).unsqueeze(0).expand(bsz, s)  # [B,S]
    base = (lengths - s).unsqueeze(1)  # [B,1]
    idx = base + pos  # [B,S]
    valid_idx = idx >= 0
    idx = idx.clamp(min=0, max=lmax - 1)

    actions_raw = items_pad.gather(1, idx)
    is_buy_raw = is_buy_pad.gather(1, idx)
    actions_raw = actions_raw.masked_fill(~valid_idx, int(old_pad_item))
    is_buy_raw = is_buy_raw.masked_fill(~valid_idx, 0)

    actions = torch.where(actions_raw == int(old_pad_item), torch.zeros_like(actions_raw), actions_raw + 1).to(
        torch.long
    )
    is_buy = is_buy_raw.to(torch.long)

    states_x = torch.zeros((bsz, s), dtype=torch.long, device=actions.device)
    states_x[:, 1:] = actions[:, :-1]

    valid_mask = actions != 0
    if bool(purchase_only):
        valid_mask = valid_mask & (is_buy == 1)
    if not bool(valid_mask.any()):
        return None

    valid_counts = valid_mask.sum(dim=1)
    keep = valid_counts > 0
    if not bool(keep.all()):
        actions = actions[keep]
        states_x = states_x[keep]
        is_buy = is_buy[keep]
        valid_mask = valid_mask[keep]
        valid_counts = valid_counts[keep]

    last_idx = (valid_counts - 1).clamp(min=0).to(torch.long)
    done_mask = torch.zeros_like(valid_mask)
    done_mask[torch.arange(int(actions.shape[0]), device=actions.device), last_idx] = True

    return {
        "states_x": states_x,
        "actions": actions,
        "is_buy": is_buy,
        "valid_mask": valid_mask,
        "done_mask": done_mask,
    }


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
        step = _make_shifted_batch_from_sessions(
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

        _, ce_logits_seq = model(states_x)
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


def _metrics_row(metrics: dict, kind: str):
    topk = metrics["topk"]
    src = metrics[kind]
    row = {}
    for k in topk:
        row[f"hr@{k}"] = float(src.get(f"hr@{k}", 0.0))
        row[f"ndcg@{k}"] = float(src.get(f"ndcg@{k}", 0.0))
    return row


def _overall_row(metrics: dict):
    topk = metrics["topk"]
    src = metrics["overall"]
    row = {}
    for k in topk:
        row[f"ndcg@{k}"] = float(src.get(f"ndcg@{k}", 0.0))
    return row


def _summary_at_k_text(val_metrics: dict, test_metrics: dict, k: int):
    def g(m: dict, section: str, key: str):
        return float(m.get(section, {}).get(key, 0.0))

    lines = [
        f"overall val/ndcg@{k}={g(val_metrics, 'overall', f'ndcg@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'overall', f'ndcg@{k}'):.6f}",
        f"click   val/hr@{k}={g(val_metrics, 'click', f'hr@{k}'):.6f} val/ndcg@{k}={g(val_metrics, 'click', f'ndcg@{k}'):.6f}  test/hr@{k}={g(test_metrics, 'click', f'hr@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'click', f'ndcg@{k}'):.6f}",
        f"purchase val/hr@{k}={g(val_metrics, 'purchase', f'hr@{k}'):.6f} val/ndcg@{k}={g(val_metrics, 'purchase', f'ndcg@{k}'):.6f}  test/hr@{k}={g(test_metrics, 'purchase', f'hr@{k}'):.6f} test/ndcg@{k}={g(test_metrics, 'purchase', f'ndcg@{k}'):.6f}",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    config_path = args.config
    cfg = _load_config(config_path)
    cfg = _apply_cli_overrides(cfg, args)
    if str(cfg.get("early_stopping_metric", "ndcg@10")) != "ndcg@10":
        raise ValueError("Only early_stopping_metric='ndcg@10' is supported.")
    reward_fn = str(cfg.get("reward_fn", "click_buy"))
    if reward_fn not in {"click_buy", "ndcg"}:
        raise ValueError("reward_fn must be one of: click_buy | ndcg")

    dataset_name = str(cfg.get("dataset", "retailrocket"))
    dataset_root = _resolve_dataset_root(dataset_name)
    config_name = Path(config_path).stem
    run_dir = _make_run_dir(dataset_name, config_name)
    _configure_logging(run_dir, debug=bool(cfg.get("debug", False)))
    _dump_config(cfg, run_dir)

    logger = logging.getLogger(__name__)
    logger.info("run_dir: %s", str(run_dir))
    logger.info("dataset: %s", dataset_name)
    if bool(getattr(args, "smoke_cpu", False)):
        logger.info("smoke_cpu: enabled (forcing CPU, batch_size=8, epoch=1, skipping val/test result file writing)")

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_epochs = int(cfg.get("epoch", 50))
    max_steps = int(cfg.get("max_steps", 0))

    smoke_cpu = bool(getattr(args, "smoke_cpu", False))
    if smoke_cpu:
        device = torch.device("cpu")
        num_epochs = 1
        train_batch_size = 8
        val_batch_size = 8
        train_num_workers = 0
        val_num_workers = 0
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{int(cfg.get('device_id', 0))}")
        else:
            device = torch.device("cpu")

    if bool(cfg.get("debug", False)) and device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    data_rel = str(cfg.get("data", "data"))
    data_directory = str(dataset_root / data_rel)
    data_statis = pd.read_pickle(os.path.join(data_directory, "data_statis.df"))
    state_size = int(data_statis["state_size"][0])
    item_num = int(data_statis["item_num"][0])
    if bool(cfg.get("debug", False)):
        logger.debug(
            "model_cfg state_size=%d hidden_factor=%d num_heads=%d item_num=%d",
            int(state_size),
            int(cfg.get("hidden_factor", 64)),
            int(cfg.get("num_heads", 1)),
            int(item_num),
        )

    reward_click = float(cfg.get("r_click", 0.2))
    reward_buy = float(cfg.get("r_buy", 1.0))
    reward_negative = float(cfg.get("r_negative", -0.0))
    purchase_only = bool(cfg.get("purchase_only", False))

    if not smoke_cpu:
        train_batch_size = int(cfg.get("batch_size_train", 256))
        val_batch_size = int(cfg.get("batch_size_val", 256))
        train_num_workers = int(cfg.get("num_workers_train", 0))
        val_num_workers = int(cfg.get("num_workers_val", 0))

    with open(os.path.join(data_directory, "pop_dict.txt"), "r") as f:
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
    pin_memory = device.type == "cuda"

    t0 = time.perf_counter()
    train_ds = _SessionDataset(data_directory=data_directory, split_df_name="sampled_train.df")
    train_ds_s = time.perf_counter() - t0
    num_sessions = int(len(train_ds))
    num_batches = int(num_sessions / train_batch_size)
    if num_batches <= 0:
        logger.warning(
            "num_batches=%d (num_sessions=%d, train_batch_size=%d) -> no training batches will run; metrics will be static",
            int(num_batches),
            int(num_sessions),
            int(train_batch_size),
        )

    t0 = time.perf_counter()
    val_ds = _SessionDataset(data_directory=data_directory, split_df_name="sampled_val.df")
    val_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    val_dl = _make_session_loader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        pad_item=item_num,
        shuffle=False,
    )
    val_dl_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_ds = _SessionDataset(data_directory=data_directory, split_df_name="sampled_test.df")
    test_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_dl = _make_session_loader(
        test_ds,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        pad_item=item_num,
        shuffle=False,
    )
    test_dl_s = time.perf_counter() - t0

    behavior_prob_table = torch.full((item_num + 1,), 1.0, dtype=torch.float32)
    for k, v in pop_dict.items():
        kk = int(k)
        if 0 <= kk < item_num:
            behavior_prob_table[kk + 1] = float(v)
    behavior_prob_table = behavior_prob_table.to(device)

    early_patience = int(cfg.get("early_stopping_ep", 5))
    best_metric = float("-inf")
    epochs_since_improve = 0
    stop_training = False

    for epoch_idx in range(num_epochs):
        if num_batches > 0:
            sampler = RandomSampler(train_ds, replacement=True, num_samples=num_batches * int(train_batch_size))
            t0 = time.perf_counter()
            dl = _make_session_loader(
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

            items_pad, is_buy_pad, lengths = batch
            step = _make_shifted_batch_from_sessions(
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

            pointer = np.random.randint(0, 2)
            if pointer == 0:
                main_qn, target_qn = qn1, qn2
                opt1, opt2 = opt1_qn1, opt2_qn1
            else:
                main_qn, target_qn = qn2, qn1
                opt1, opt2 = opt1_qn2, opt2_qn2

            main_qn.train()
            target_qn.train()

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
            action_flat = actions[valid_mask]
            is_buy_flat = is_buy[valid_mask]
            done_flat = done_mask[valid_mask].to(torch.float32)

            if reward_fn == "ndcg":
                with torch.no_grad():
                    reward_flat = _ndcg_reward_from_logits(ce_flat.detach(), action_flat)
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
            neg_actions = _sample_negative_actions(1, item_num + 1, action_flat, int(cfg.get("neg", 10)), device=device)
            q_sneg = q_curr_flat.gather(1, neg_actions)
            qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

            ce_loss_pre = F.cross_entropy(ce_flat, action_flat, reduction="none")

            if total_step < 15000:
                loss = qloss_pos + qloss_neg + ce_loss_pre.mean()
                if bool(cfg.get("debug", False)) and (not torch.isfinite(loss).all()):
                    raise FloatingPointError(f"Non-finite loss (phase1) at total_step={int(total_step)}")
                opt1.zero_grad(set_to_none=True)
                loss.backward()
                opt1.step()
                total_step += int(step_count)
            else:
                with torch.no_grad():
                    prob = F.softmax(ce_flat, dim=1).gather(1, action_flat[:, None]).squeeze(1)
                behavior_prob = behavior_prob_table[action_flat]
                ips = (prob / behavior_prob).clamp(0.1, 10.0).pow(float(cfg.get("smooth", 0.0)))

                with torch.no_grad():
                    q_pos_det = q_curr_flat.gather(1, action_flat[:, None]).squeeze(1)
                    q_neg_det = q_curr_flat.gather(1, neg_actions).sum(dim=1)
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
                total_step += int(step_count)

        val_metrics = evaluate(
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

    best_model = SASRecQNetworkRectools(
        item_num=item_num,
        state_size=state_size,
        hidden_size=int(cfg.get("hidden_factor", 64)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_blocks=int(cfg.get("num_blocks", 1)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    ).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    val_best = evaluate(
        best_model,
        val_dl,
        reward_click,
        reward_buy,
        device,
        debug=bool(cfg.get("debug", False)),
        split="val(best)",
        state_size=state_size,
        item_num=item_num,
        purchase_only=purchase_only,
    )
    test_best = evaluate(
        best_model,
        test_dl,
        reward_click,
        reward_buy,
        device,
        debug=bool(cfg.get("debug", False)),
        split="test(best)",
        state_size=state_size,
        item_num=item_num,
        purchase_only=purchase_only,
    )

    val_click = _metrics_row(val_best, "click")
    test_click = _metrics_row(test_best, "click")
    val_purchase = _metrics_row(val_best, "purchase")
    test_purchase = _metrics_row(test_best, "purchase")
    val_overall = _overall_row(val_best)
    test_overall = _overall_row(test_best)

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

    if not smoke_cpu:
        df_clicks = pd.DataFrame([click_row], index=["metrics"]).loc[:, col_order]
        df_purchase = pd.DataFrame([purchase_row], index=["metrics"]).loc[:, col_order]
        df_clicks.to_csv(run_dir / "results_clicks.csv", index=False)
        df_purchase.to_csv(run_dir / "results_purchase.csv", index=False)

    overall_col_order = []
    for k in val_best["topk"]:
        overall_col_order.extend([f"val/ndcg@{k}", f"test/ndcg@{k}"])
    overall_row = {}
    for k in val_best["topk"]:
        overall_row[f"val/ndcg@{k}"] = float(val_overall.get(f"ndcg@{k}", 0.0))
        overall_row[f"test/ndcg@{k}"] = float(test_overall.get(f"ndcg@{k}", 0.0))
    if not smoke_cpu:
        df_overall = pd.DataFrame([overall_row], index=["metrics"]).loc[:, overall_col_order]
        df_overall.to_csv(run_dir / "results.csv", index=False)

        with open(run_dir / "summary@10.txt", "w") as f:
            f.write(_summary_at_k_text(val_best, test_best, k=10))


if __name__ == "__main__":
    main()


