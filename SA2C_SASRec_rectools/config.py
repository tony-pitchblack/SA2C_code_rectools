from __future__ import annotations

import numpy as np
import yaml


def default_config() -> dict:
    return {
        "model_type": "sasrec",
        "trainer": None,
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
        "crr": {
            "temperature": 1.0,
            "weight_type": "exp",
            "advantage_baseline": "mean",
            "tau": 0.005,
            "critic_loss_weight": 1.0,
            "actor_lr": None,
            "critic_lr": None,
            "gamma": 0.5,
        },
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
        "pretrained_backbone": {
            "use": False,
            "pretrained_config_name": None,
            "backbone_lr": None,
            "backbone_lr_2": None,
        },
        "discount": 0.5,
        "neg": 10,
        # Backward-compat alias for num_val_negative_samples (eval candidate pool size for bert4rec_loo).
        "val_samples_num": None,
        # Bert4recv1-style: number of eval negatives (popular items) used as candidates.
        # - null -> full vocabulary candidates
        # - int  -> use top-K popular items
        # - float in (0, 1] -> use ceil(vocab * pct) popular items
        "num_val_negative_samples": None,
        "ce_n_negatives": None,
        "sampled_loss": {
            "use": False,
            "ce_n_negatives": 256,
            "critic_n_negatives": 256,
        },
        "actor": {},
        "critic": {"type": "full-vocab"},
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
        "albert4rec": {
            "masking_proba": 0.2,
            "n_negatives": 256,
            "intermediate_size": None,
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


def _resolve_ce_n_negatives_cfg(cfg: dict):
    if "ce_n_negatives" in cfg:
        v = cfg.get("ce_n_negatives", None)
        if v is not None:
            return v
    sampled_cfg = cfg.get("sampled_loss") or {}
    if isinstance(sampled_cfg, dict) and ("ce_n_negatives" in sampled_cfg):
        return sampled_cfg.get("ce_n_negatives", None)
    return None


def resolve_ce_sampling(*, cfg: dict, item_num: int) -> tuple[int, int, float | None, int | None]:
    full_vocab = int(item_num)
    raw = _resolve_ce_n_negatives_cfg(cfg)
    if raw is None:
        return full_vocab, full_vocab, None, None
    if isinstance(raw, bool):
        return full_vocab, full_vocab, None, None
    try:
        if isinstance(raw, int):
            nneg = int(raw)
            if nneg <= 0:
                return 1, full_vocab, None, 0
            return 1 + nneg, full_vocab, None, nneg
        x = float(raw)
    except Exception:
        return full_vocab, full_vocab, None, None

    if x >= 1.0:
        return full_vocab, full_vocab, None, None
    if x <= 0.0:
        return 1, full_vocab, 0.0, 0
    nneg = int(full_vocab * x)
    if nneg <= 0:
        return 1, full_vocab, float(x), 0
    return 1 + nneg, full_vocab, float(x), nneg


def resolve_num_val_negative_samples(*, cfg: dict, item_num: int) -> tuple[int | None, float | None]:
    """
    Bert4recv1-style eval candidate pool size (shared negatives list):
    - None -> full vocab
    - int  -> top-K popular items
    - float in (0,1] -> ceil(item_num * pct)
    """
    raw = cfg.get("num_val_negative_samples", None)
    if raw is None and ("val_samples_num" in cfg):
        raw = cfg.get("val_samples_num", None)
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, int):
        k = int(raw)
        if k < 0:
            raise ValueError("num_val_negative_samples must be null or a non-negative int/float")
        return k, None
    try:
        x = float(raw)
    except Exception as e:
        raise ValueError("num_val_negative_samples must be null or a non-negative int/float") from e
    if x < 0.0:
        raise ValueError("num_val_negative_samples must be null or a non-negative int/float")
    if x == 0.0:
        return 0, 0.0
    if 0.0 < x <= 1.0:
        k = int(np.ceil(float(item_num) * float(x)))
        return max(0, min(int(k), int(item_num))), float(x)
    if float(x).is_integer():
        k = int(x)
        return max(0, min(int(k), int(item_num))), None
    raise ValueError("num_val_negative_samples as float must be in (0,1] or an integer-valued float")


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


def _validate_optional_lstm_block(block, *, prefix: str) -> dict | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"{prefix} must be a mapping (dict) or null")
    for k in ("hidden_size", "num_layers", "dropout_rate"):
        if k not in block:
            raise ValueError(f"Missing required config: {prefix}.{k}")
    hidden_size = int(block.get("hidden_size"))
    num_layers = int(block.get("num_layers"))
    dropout_rate = float(block.get("dropout_rate"))
    if hidden_size <= 0:
        raise ValueError(f"{prefix}.hidden_size must be > 0")
    if num_layers <= 0:
        raise ValueError(f"{prefix}.num_layers must be > 0")
    if dropout_rate < 0.0:
        raise ValueError(f"{prefix}.dropout_rate must be >= 0")
    return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout_rate": dropout_rate}


def _validate_optional_state_mlp_block(block, *, prefix: str) -> dict | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"{prefix} must be a mapping (dict) or null")
    if "hidden_sizes" not in block:
        raise ValueError(f"Missing required config: {prefix}.hidden_sizes")
    if "dropout_rate" not in block:
        raise ValueError(f"Missing required config: {prefix}.dropout_rate")
    hidden_sizes = block.get("hidden_sizes")
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0 or not all(isinstance(x, int) for x in hidden_sizes):
        raise ValueError(f"{prefix}.hidden_sizes must be a non-empty list of ints")
    dropout_rate = float(block.get("dropout_rate"))
    if dropout_rate < 0.0:
        raise ValueError(f"{prefix}.dropout_rate must be >= 0")
    return {"hidden_sizes": [int(x) for x in hidden_sizes], "dropout_rate": float(dropout_rate)}


def validate_crr_actor_cfg(cfg: dict) -> tuple[dict | None, dict | None]:
    actor_cfg = cfg.get("actor") or {}
    if not isinstance(actor_cfg, dict):
        raise ValueError("actor must be a mapping (dict)")
    actor_lstm_cfg = None
    if "lstm" in actor_cfg:
        actor_lstm_cfg = _validate_optional_lstm_block(actor_cfg.get("lstm"), prefix="actor.lstm")
    actor_mlp_cfg = None
    if "mlp" in actor_cfg:
        actor_mlp_cfg = _validate_optional_state_mlp_block(actor_cfg.get("mlp"), prefix="actor.mlp")
    return actor_lstm_cfg, actor_mlp_cfg


def validate_crr_critic_cfg(cfg: dict) -> tuple[str, dict | None, dict | None]:
    critic_cfg = cfg.get("critic") or {}
    if not isinstance(critic_cfg, dict):
        raise ValueError("critic must be a mapping (dict)")
    critic_type = str(critic_cfg.get("type", "full-vocab")).strip().lower()
    if critic_type not in {"pointwise", "full-vocab"}:
        raise ValueError("critic.type must be one of: pointwise | full-vocab")
    critic_lstm_cfg = None
    if "lstm" in critic_cfg:
        critic_lstm_cfg = _validate_optional_lstm_block(critic_cfg.get("lstm"), prefix="critic.lstm")
    critic_mlp_cfg = None
    if "mlp" in critic_cfg:
        critic_mlp_cfg = _validate_optional_state_mlp_block(critic_cfg.get("mlp"), prefix="critic.mlp")
    return critic_type, critic_lstm_cfg, critic_mlp_cfg


def resolve_trainer(cfg: dict) -> str:
    raw = cfg.get("trainer", None)
    if raw is None:
        enable_sa2c = bool(cfg.get("enable_sa2c", True))
        return "sa2c" if enable_sa2c else "baseline"
    s = str(raw).strip().lower()
    if s not in {"baseline", "sa2c", "crr"}:
        raise ValueError("trainer must be one of: baseline | sa2c | crr | null")
    return s


__all__ = [
    "default_config",
    "load_config",
    "apply_cli_overrides",
    "is_persrec_tc5_dataset_cfg",
    "validate_crr_actor_cfg",
    "validate_crr_critic_cfg",
    "validate_pointwise_critic_cfg",
    "resolve_ce_sampling",
    "resolve_num_val_negative_samples",
    "resolve_trainer",
]

