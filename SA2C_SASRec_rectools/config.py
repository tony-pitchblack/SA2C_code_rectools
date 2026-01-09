from __future__ import annotations

import yaml


def default_config() -> dict:
    return {
        "seed": 0,
        "epoch": 50,
        "dataset": "retailrocket",
        "data": "data",
        "sanity": False,
        "purchase_only": False,
        "reward_fn": "click_buy",
        "enable_sa2c": True,
        "warmup_epochs": 0.02,
        "early_stopping_warmup_ep": None,
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
        "sampled_loss": {
            "use": False,
            "ce_n_negatives": 256,
            "critic_n_negatives": 256,
        },
        "bert4rec_loo": {
            "enable": False,
            "val_samples_num": 0,
            "test_samples_num": 0,
        },
        "weight": 1.0,
        "smooth": 0.0,
        "clip": 0.0,
        "max_steps": 0,
        "debug": False,
        "early_stopping_ep": 5,
        "early_stopping_metric": "ndcg@10",
    }


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    cfg = default_config()
    cfg.update(data)
    return cfg


def apply_cli_overrides(cfg: dict, args) -> dict:
    sanity_cli = bool(getattr(args, "sanity", False))
    dataset_cfg = cfg.get("dataset", None)
    sanity_cfg = bool(cfg.get("sanity", False))
    sanity_dataset = bool(dataset_cfg.get("use_sanity_subset", False)) if isinstance(dataset_cfg, dict) else False
    sanity = bool(sanity_cli or sanity_cfg or sanity_dataset)
    cfg["sanity"] = sanity
    if isinstance(dataset_cfg, dict) and ("use_sanity_subset" in dataset_cfg):
        dataset_cfg["use_sanity_subset"] = bool(sanity)

    if args.early_stopping_ep is not None:
        cfg["early_stopping_ep"] = int(args.early_stopping_ep)
    if args.early_stopping_metric is not None:
        cfg["early_stopping_metric"] = str(args.early_stopping_metric)
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if bool(args.debug):
        cfg["debug"] = True
    return cfg


def is_persrec_tc5_dataset_cfg(dataset_cfg) -> bool:
    return isinstance(dataset_cfg, dict) and ("calc_date" in dataset_cfg)


__all__ = ["default_config", "load_config", "apply_cli_overrides", "is_persrec_tc5_dataset_cfg"]

