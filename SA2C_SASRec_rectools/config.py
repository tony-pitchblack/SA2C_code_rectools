from __future__ import annotations

import yaml


def default_config() -> dict:
    return {
        "gridsearch": {
            "enable": False,
            "metric": "overall.ndcg@10",
            "epochs_per_run": 5,
            "direction": "maximize",
            "n_trials": 20,
            "timeout_s": 0,
            "n_jobs": 1,
            "seed": 0,
            "n_startup_trials": 10,
            "pruner": {"enable": True, "n_warmup_epochs": 2},
            "allow_early_stopping": False,
            "max_steps_per_run": 0,
        },
        "seed": 0,
        "epoch": 50,
        "dataset": "retailrocket",
        "data": "data",
        "sanity": False,
        "limit_chunks_pct": None,
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
        "pointwise_critic": {
            "use": False,
            "arch": "dot",
            "mlp": {
                "hidden_sizes": [64],
                "dropout_rate": 0.0,
            },
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


def validate_pointwise_critic_cfg(cfg: dict) -> tuple[bool, str, dict | None]:
    pointwise_cfg = cfg.get("pointwise_critic") or {}
    if not isinstance(pointwise_cfg, dict):
        raise ValueError("pointwise_critic must be a mapping (dict)")
    use = bool(pointwise_cfg.get("use", False))
    arch = str(pointwise_cfg.get("arch", "dot"))
    if arch not in {"dot", "mlp"}:
        raise ValueError("pointwise_critic.arch must be one of: dot | mlp")
    if arch != "mlp":
        return use, arch, None

    mlp_cfg = pointwise_cfg.get("mlp", None)
    if not isinstance(mlp_cfg, dict):
        raise ValueError("pointwise_critic.mlp must be provided when pointwise_critic.arch=mlp")

    if "hidden_sizes" not in mlp_cfg:
        raise ValueError("Missing required config: pointwise_critic.mlp.hidden_sizes")
    if "dropout_rate" not in mlp_cfg:
        raise ValueError("Missing required config: pointwise_critic.mlp.dropout_rate")

    hidden_sizes = mlp_cfg.get("hidden_sizes")
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0 or not all(isinstance(x, int) for x in hidden_sizes):
        raise ValueError("pointwise_critic.mlp.hidden_sizes must be a non-empty list of ints")
    dropout_rate = mlp_cfg.get("dropout_rate")
    try:
        dropout_rate_f = float(dropout_rate)
    except Exception as e:
        raise ValueError("pointwise_critic.mlp.dropout_rate must be a float") from e

    return use, arch, {"hidden_sizes": [int(x) for x in hidden_sizes], "dropout_rate": float(dropout_rate_f)}


__all__ = ["default_config", "load_config", "apply_cli_overrides", "is_persrec_tc5_dataset_cfg", "validate_pointwise_critic_cfg"]

