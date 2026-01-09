from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from .logging_utils import configure_logging, dump_config
from .metrics import evaluate, get_metric_value
from .models import SASRecBaselineRectools, SASRecQNetworkRectools
from .optuna_dsl import apply_optuna_suggestions
from .training.baseline import train_baseline
from .training.sa2c import train_sa2c


def run_optuna_gridsearch(
    *,
    cfg: dict,
    base_run_dir: Path,
    device: torch.device,
    train_ds,
    val_dl,
    test_dl,
    pop_dict_path: Path | None,
    reward_click: float,
    reward_buy: float,
    reward_negative: float,
    state_size: int,
    item_num: int,
    purchase_only: bool,
    num_batches: int,
    train_batch_size: int,
    train_num_workers: int,
    pin_memory: bool,
    reward_fn: str,
    smoke_cpu: bool,
):
    try:
        import optuna
    except Exception as e:  # pragma: no cover
        raise RuntimeError("optuna is required when gridsearch.enable=true") from e

    gs = dict(cfg.get("gridsearch") or {})
    metric_key = str(gs.get("metric", "overall.ndcg@10"))
    direction = str(gs.get("direction", "maximize"))
    if direction not in {"maximize", "minimize"}:
        raise ValueError("gridsearch.direction must be maximize|minimize")

    n_jobs = int(gs.get("n_jobs", 1))
    if n_jobs != 1:
        raise ValueError("gridsearch.n_jobs != 1 is not supported (logging/checkpointing are not trial-isolated)")

    n_trials = int(gs.get("n_trials", 20))
    timeout_s = int(gs.get("timeout_s", 0))
    timeout = None if timeout_s <= 0 else float(timeout_s)
    seed = int(gs.get("seed", int(cfg.get("seed", 0))))
    n_startup_trials = int(gs.get("n_startup_trials", 10))
    epochs_per_run = int(gs.get("epochs_per_run", 5))
    allow_early_stopping = bool(gs.get("allow_early_stopping", False))
    max_steps_per_run = int(gs.get("max_steps_per_run", 0))

    pr_cfg = dict(gs.get("pruner") or {})
    enable_pruner = bool(pr_cfg.get("enable", True))
    n_warmup_epochs = int(pr_cfg.get("n_warmup_epochs", 2))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(0, n_warmup_epochs)) if enable_pruner else optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(0, n_startup_trials))

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    logger = logging.getLogger(__name__)

    gs_dir = base_run_dir / "gridsearch"
    gs_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = apply_optuna_suggestions(cfg, trial)
        if not allow_early_stopping:
            trial_cfg["early_stopping_ep"] = 0
        if max_steps_per_run > 0:
            trial_cfg["max_steps"] = int(max_steps_per_run)

        trial_run_dir = gs_dir / f"trial_{int(trial.number):04d}"
        trial_run_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(trial_run_dir, debug=bool(trial_cfg.get("debug", False)))
        dump_config(trial_cfg, trial_run_dir)

        enable_sa2c = bool(trial_cfg.get("enable_sa2c", True))
        num_epochs = int(epochs_per_run)
        max_steps = int(trial_cfg.get("max_steps", 0))

        try:
            if enable_sa2c:
                reward_negative_trial = float(trial_cfg.get("r_negative", reward_negative))
                if pop_dict_path is None:
                    raise ValueError("pop_dict_path is required when enable_sa2c=true")
                best_path, _ = train_sa2c(
                    cfg=trial_cfg,
                    train_ds=train_ds,
                    val_dl=val_dl,
                    pop_dict_path=Path(pop_dict_path),
                    run_dir=trial_run_dir,
                    device=device,
                    reward_click=reward_click,
                    reward_buy=reward_buy,
                    reward_negative=reward_negative_trial,
                    state_size=state_size,
                    item_num=item_num,
                    purchase_only=purchase_only,
                    num_epochs=num_epochs,
                    num_batches=num_batches,
                    train_batch_size=train_batch_size,
                    train_num_workers=train_num_workers,
                    pin_memory=pin_memory,
                    max_steps=max_steps,
                    reward_fn=reward_fn,
                    metric_key=metric_key,
                    trial=trial,
                )
                model = SASRecQNetworkRectools(
                    item_num=item_num,
                    state_size=state_size,
                    hidden_size=int(trial_cfg.get("hidden_factor", 64)),
                    num_heads=int(trial_cfg.get("num_heads", 1)),
                    num_blocks=int(trial_cfg.get("num_blocks", 1)),
                    dropout_rate=float(trial_cfg.get("dropout_rate", 0.1)),
                ).to(device)
            else:
                best_path = train_baseline(
                    cfg=trial_cfg,
                    train_ds=train_ds,
                    val_dl=val_dl,
                    run_dir=trial_run_dir,
                    device=device,
                    reward_click=reward_click,
                    reward_buy=reward_buy,
                    state_size=state_size,
                    item_num=item_num,
                    purchase_only=purchase_only,
                    num_epochs=num_epochs,
                    num_batches=num_batches,
                    train_batch_size=train_batch_size,
                    train_num_workers=train_num_workers,
                    pin_memory=pin_memory,
                    max_steps=max_steps,
                    metric_key=metric_key,
                    trial=trial,
                )
                model = SASRecBaselineRectools(
                    item_num=item_num,
                    state_size=state_size,
                    hidden_size=int(trial_cfg.get("hidden_factor", 64)),
                    num_heads=int(trial_cfg.get("num_heads", 1)),
                    num_blocks=int(trial_cfg.get("num_blocks", 1)),
                    dropout_rate=float(trial_cfg.get("dropout_rate", 0.1)),
                ).to(device)

            model.load_state_dict(torch.load(best_path, map_location=device))
            val_best = evaluate(
                model,
                val_dl,
                reward_click,
                reward_buy,
                device,
                debug=bool(trial_cfg.get("debug", False)),
                split="val(best)",
                state_size=state_size,
                item_num=item_num,
                purchase_only=purchase_only,
            )
            score = float(get_metric_value(val_best, metric_key))
            trial.set_user_attr("best_model_path", str(best_path))
            trial.set_user_attr("trial_run_dir", str(trial_run_dir))
            return score
        except RuntimeError as e:
            if str(e) == "optuna_pruned":
                raise optuna.TrialPruned()
            raise

    study.optimize(objective, n_trials=max(0, n_trials), timeout=timeout, gc_after_trial=True)

    best = study.best_trial
    best_dir = base_run_dir / "gridsearch_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    with open(best_dir / "best_trial.yml", "w") as f:
        yaml.safe_dump(
            {
                "value": float(best.value),
                "params": dict(best.params),
                "metric": metric_key,
                "direction": direction,
                "trial_run_dir": best.user_attrs.get("trial_run_dir"),
                "best_model_path": best.user_attrs.get("best_model_path"),
            },
            f,
            sort_keys=False,
        )

    if not smoke_cpu and best.user_attrs.get("best_model_path") and best.user_attrs.get("trial_run_dir"):
        best_model_path = Path(str(best.user_attrs["best_model_path"]))
        trial_run_dir = Path(str(best.user_attrs["trial_run_dir"]))
        best_cfg_path = trial_run_dir / "config.yml"
        if best_cfg_path.exists():
            best_cfg = yaml.safe_load(best_cfg_path.read_text()) or {}
        else:
            best_cfg = deepcopy(cfg)
        dump_config(best_cfg, best_dir)

        enable_sa2c = bool(best_cfg.get("enable_sa2c", True))
        if enable_sa2c:
            best_model = SASRecQNetworkRectools(
                item_num=item_num,
                state_size=state_size,
                hidden_size=int(best_cfg.get("hidden_factor", 64)),
                num_heads=int(best_cfg.get("num_heads", 1)),
                num_blocks=int(best_cfg.get("num_blocks", 1)),
                dropout_rate=float(best_cfg.get("dropout_rate", 0.1)),
            ).to(device)
        else:
            best_model = SASRecBaselineRectools(
                item_num=item_num,
                state_size=state_size,
                hidden_size=int(best_cfg.get("hidden_factor", 64)),
                num_heads=int(best_cfg.get("num_heads", 1)),
                num_blocks=int(best_cfg.get("num_blocks", 1)),
                dropout_rate=float(best_cfg.get("dropout_rate", 0.1)),
            ).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        val_best = evaluate(
            best_model,
            val_dl,
            reward_click,
            reward_buy,
            device,
            debug=bool(best_cfg.get("debug", False)),
            split="val(best_gridsearch)",
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
            debug=bool(best_cfg.get("debug", False)),
            split="test(best_gridsearch)",
            state_size=state_size,
            item_num=item_num,
            purchase_only=purchase_only,
        )
        from .artifacts import write_results

        write_results(run_dir=best_dir, val_best=val_best, test_best=test_best, smoke_cpu=smoke_cpu)

    logger.info("gridsearch complete: best_value=%f best_params=%s", float(best.value), dict(best.params))
    return study


__all__ = ["run_optuna_gridsearch"]

