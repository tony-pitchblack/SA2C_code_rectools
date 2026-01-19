from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from .artifacts import write_results
from .cli import parse_args
from .config import (
    apply_cli_overrides,
    is_persrec_tc5_dataset_cfg,
    load_config,
    resolve_ce_sampling,
    resolve_num_val_negative_samples,
    resolve_trainer,
    validate_crr_actor_cfg,
    validate_crr_critic_cfg,
    validate_pointwise_critic_cfg,
)
from .data_utils.bert4rec_loo import prepare_persrec_tc5_bert4rec_loo, prepare_sessions_bert4rec_loo
from .data_utils.persrec_tc5 import prepare_persrec_tc5
from .data_utils.albert4rec import make_albert4rec_loader
from .data_utils.sessions import SessionDataset, make_session_loader
from .distributed import (
    barrier,
    ddp_cleanup,
    ddp_setup,
    find_free_port,
    is_distributed,
    is_rank0,
    parse_cuda_devices,
    silence_logging_if_needed,
)
from .logging_utils import configure_logging, dump_config
from .models import Albert4Rec, SASRecBaselineRectools, SASRecQNetworkRectools
from .paths import make_run_dir, resolve_dataset_root
from .metrics import evaluate, evaluate_albert4rec_loo, evaluate_loo, evaluate_loo_candidates
from .gridsearch import run_optuna_gridsearch
from .training.albert4rec import train_albert4rec
from .training.baseline import train_baseline
from .training.crr import train_crr
from .training.sa2c import train_sa2c


def _infer_eval_scheme_from_config_path(config_path: str, *, dataset_name: str) -> str | None:
    p = Path(config_path)
    parent_parts = list(p.parent.parts)
    for i in range(len(parent_parts) - 1, -1, -1):
        if parent_parts[i] == str(dataset_name):
            if i < len(parent_parts) - 1:
                return str(parent_parts[i + 1])
            return None
    return None


def _select_device(*, cfg: dict, smoke_cpu: bool) -> torch.device:
    if bool(smoke_cpu):
        return torch.device("cpu")
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{int(torch.cuda.current_device())}")
        dev = cfg.get("device_id", None)
        if isinstance(dev, int):
            return torch.device(f"cuda:{int(dev)}")
        if isinstance(dev, str):
            s = dev.strip()
            if s.startswith("cuda"):
                return torch.device(s)
        return torch.device("cuda")
    return torch.device("cpu")


def _worker_main(
    *,
    cfg: dict,
    args,
    local_rank: int,
    world_size: int,
    device_ids: list[int] | None,
) -> None:
    silence_logging_if_needed(is_rank0=is_rank0())
    eval_only = bool(getattr(args, "eval_only", False))
    if bool(eval_only) and is_distributed() and (not is_rank0()):
        barrier()
        return
    continue_training = bool(getattr(args, "continue_training", False))
    config_path = args.config

    model_type = str(cfg.get("model_type", "sasrec")).strip().lower()
    if model_type not in {"sasrec", "albert4rec"}:
        raise ValueError("model_type must be one of: sasrec | albert4rec")

    if str(cfg.get("early_stopping_metric", "ndcg@10")) != "ndcg@10":
        raise ValueError("Only early_stopping_metric='ndcg@10' is supported.")
    reward_fn = str(cfg.get("reward_fn", "click_buy"))
    if reward_fn not in {"click_buy", "ndcg"}:
        raise ValueError("reward_fn must be one of: click_buy | ndcg")
    trainer = resolve_trainer(cfg)
    enable_sa2c = trainer in {"sa2c", "crr"}
    pointwise_critic_use = False
    pointwise_critic_arch = "dot"
    pointwise_mlp_cfg = None
    actor_lstm_cfg = None
    actor_mlp_cfg = None
    critic_lstm_cfg = None
    critic_mlp_cfg = None
    if trainer == "crr":
        actor_lstm_cfg, actor_mlp_cfg = validate_crr_actor_cfg(cfg)
        critic_type, critic_lstm_cfg, critic_mlp_cfg = validate_crr_critic_cfg(cfg)
        pointwise_critic_use = str(critic_type) == "pointwise"
        pointwise_critic_arch = "dot"
        pointwise_mlp_cfg = None
    elif trainer == "sa2c":
        pointwise_critic_use, pointwise_critic_arch, pointwise_mlp_cfg = validate_pointwise_critic_cfg(cfg)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = cfg.get("dataset", "retailrocket")
    persrec_tc5 = is_persrec_tc5_dataset_cfg(dataset_cfg)
    plu_filter_raw = getattr(args, "plu_filter", None)
    if persrec_tc5:
        plu_filter_mode = "enable" if plu_filter_raw is None else str(plu_filter_raw)
    else:
        if plu_filter_raw is not None:
            raise ValueError("--plu-filter is supported only for persrec_tc5 datasets")
        plu_filter_mode = None
    if persrec_tc5:
        calc_date = str(dataset_cfg.get("calc_date"))
        dataset_name = f"persrec_tc5_{calc_date}"
        dataset_root = repo_root
    else:
        dataset_name = str(dataset_cfg)
        dataset_root = resolve_dataset_root(dataset_name)

    config_p = Path(config_path).resolve()
    logs_root = (repo_root / "logs" / "SA2C_SASRec_rectools").resolve()
    use_run_dir_from_config = (
        (eval_only or continue_training)
        and config_p.name in {"config.yml", "config.yaml"}
        and logs_root in config_p.parents
    )
    if use_run_dir_from_config:
        config_name = config_p.parent.name
    else:
        config_name = Path(config_path).stem
        if bool(getattr(args, "sanity", False)):
            config_name = f"{config_name}_sanity"
    eval_scheme = _infer_eval_scheme_from_config_path(config_path, dataset_name=dataset_name)
    if persrec_tc5 and eval_scheme == "bert4rec_eval" and plu_filter_mode in {"disable", "inverse"}:
        eval_scheme = f"bert4rec_eval_plu-{'disable' if plu_filter_mode == 'disable' else 'inverse'}"
    run_dir = make_run_dir(dataset_name, config_name, eval_scheme=eval_scheme)

    pretrained_backbone_cfg = cfg.get("pretrained_backbone") or {}
    if not isinstance(pretrained_backbone_cfg, dict):
        raise ValueError("pretrained_backbone must be a mapping (dict)")
    use_pretrained_backbone = bool(pretrained_backbone_cfg.get("use", False))
    if use_pretrained_backbone and (not enable_sa2c):
        raise ValueError("pretrained_backbone.use=true requires enable_sa2c=true")
    if use_pretrained_backbone:
        if "pretrained_config_name" not in pretrained_backbone_cfg:
            raise ValueError("Missing required config: pretrained_backbone.pretrained_config_name")
        if "backbone_lr" not in pretrained_backbone_cfg:
            raise ValueError("Missing required config: pretrained_backbone.backbone_lr")
        if "backbone_lr_2" not in pretrained_backbone_cfg:
            raise ValueError("Missing required config: pretrained_backbone.backbone_lr_2")

        pretrained_config_name = pretrained_backbone_cfg.get("pretrained_config_name")
        if not isinstance(pretrained_config_name, str) or (not pretrained_config_name.strip()):
            raise ValueError("pretrained_backbone.pretrained_config_name must be a non-empty string")

        for k in ("backbone_lr", "backbone_lr_2"):
            v = pretrained_backbone_cfg.get(k, None)
            if v is None:
                continue
            try:
                pretrained_backbone_cfg[k] = float(v)
            except Exception as e:
                raise ValueError(f"pretrained_backbone.{k} must be a float or null") from e
        cfg["pretrained_backbone"] = pretrained_backbone_cfg

    if is_rank0():
        configure_logging(run_dir, debug=bool(cfg.get("debug", False)))
        dump_config(cfg, run_dir)

    logger = logging.getLogger(__name__)
    logger.info("run_dir: %s", str(run_dir))
    logger.info("dataset: %s", dataset_name)
    if bool(continue_training) and is_rank0():
        logger.info(
            "continue: enabled; will resume SA2C from run_dir checkpoints if present (expected: %s, %s)",
            str(run_dir / "best_model.pt"),
            str(run_dir / "best_model_warmup.pt"),
        )
    if bool(getattr(args, "smoke_cpu", False)):
        logger.info("smoke_cpu: enabled (forcing CPU, batch_size=8, epoch=1, skipping val/test result file writing)")

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bert4rec_loo_cfg = cfg.get("bert4rec_loo") or {}
    use_bert4rec_loo = bool(isinstance(bert4rec_loo_cfg, dict) and bool(bert4rec_loo_cfg.get("enable", False)))
    val_split_samples_num = int(bert4rec_loo_cfg.get("val_samples_num", 0)) if use_bert4rec_loo else 0
    test_split_samples_num = int(bert4rec_loo_cfg.get("test_samples_num", 0)) if use_bert4rec_loo else 0
    sanity = bool(getattr(args, "sanity", False)) or bool(cfg.get("sanity", False))
    if use_bert4rec_loo and sanity:
        cap = 1000
        val_split_samples_num = min(int(val_split_samples_num), int(cap))
        test_split_samples_num = min(int(test_split_samples_num), int(cap))
    if model_type == "albert4rec":
        if not bool(use_bert4rec_loo):
            raise ValueError("albert4rec is supported only with bert4rec_loo.enable=true (bert4rec_eval)")
        if bool(enable_sa2c):
            raise ValueError("albert4rec requires enable_sa2c=false")

    for k in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
        v = cfg.get(k, None)
        if v is not None and v not in (0, 0.0, "0", "0.0"):
            raise ValueError(f"{k} is no longer supported; use limit_chunks_pct (persrec_tc5 only)")

    limit_chunks_pct_cfg = cfg.get("limit_chunks_pct", None)
    limit_chunks_pct = None
    if limit_chunks_pct_cfg is not None and limit_chunks_pct_cfg not in (0, 0.0, "0", "0.0"):
        try:
            limit_chunks_pct = float(limit_chunks_pct_cfg)
        except Exception as e:
            raise ValueError("limit_chunks_pct must be a float in [0, 1]") from e
        if not (0.0 < float(limit_chunks_pct) <= 1.0):
            raise ValueError("limit_chunks_pct must be a float in (0, 1]")
        if (not persrec_tc5) and (not use_bert4rec_loo):
            raise ValueError("limit_chunks_pct for sessions datasets requires bert4rec_loo.enable=true")
        if bool(sanity):
            raise ValueError("limit_chunks_pct cannot be used together with --sanity")

    num_epochs = int(cfg.get("epoch", 50))
    max_steps = int(cfg.get("max_steps", 0))

    smoke_cpu = bool(getattr(args, "smoke_cpu", False))
    if smoke_cpu and is_distributed():
        raise ValueError("--smoke-cpu is not supported with DDP (WORLD_SIZE>1)")
    if smoke_cpu:
        num_epochs = 1
        train_batch_size = 8
        val_batch_size = 8
        train_num_workers = 0
        val_num_workers = 0
    device = _select_device(cfg=cfg, smoke_cpu=smoke_cpu)
    if device.type == "cuda" and (not is_distributed()) and device.index is not None:
        torch.cuda.set_device(int(device.index))

    if bool(cfg.get("debug", False)) and device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    data_rel = str(cfg.get("data", "data"))
    if persrec_tc5:
        if use_bert4rec_loo:
            data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds = prepare_persrec_tc5_bert4rec_loo(
                dataset_root=dataset_root,
                data_rel=data_rel,
                dataset_name=dataset_name,
                dataset_cfg=dict(dataset_cfg),
                seed=int(cfg.get("seed", 0)),
                val_samples_num=int(val_split_samples_num),
                test_samples_num=int(test_split_samples_num),
                plu_filter=str(plu_filter_mode),
                limit_chunks_pct=limit_chunks_pct,
            )
        else:
            data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds = prepare_persrec_tc5(
                dataset_root=dataset_root,
                data_rel=data_rel,
                dataset_name=dataset_name,
                dataset_cfg=dict(dataset_cfg),
                seed=int(cfg.get("seed", 0)),
                limit_chunks_pct=limit_chunks_pct,
            )
    else:
        data_directory = str(dataset_root / data_rel)
        data_statis_path = Path(data_directory) / "data_statis.df"
        pop_dict_path = Path(data_directory) / "pop_dict.txt"
        train_ds = None
        val_ds = None
        test_ds = None

    data_statis = pd.read_pickle(str(data_statis_path))
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

    ce_loss_vocab_size, ce_full_vocab_size, ce_vocab_pct, _ = resolve_ce_sampling(cfg=cfg, item_num=item_num)

    eval_neg_samples_num, eval_neg_vocab_pct = resolve_num_val_negative_samples(cfg=cfg, item_num=item_num)

    sampled_negatives = None
    if use_bert4rec_loo and model_type != "albert4rec":
        if eval_neg_samples_num is None:
            sampled_negatives = torch.arange(1, int(item_num) + 1, device=device, dtype=torch.long)
        else:
            with open(str(pop_dict_path), "r") as f:
                pop_dict = eval(f.read())
            if not isinstance(pop_dict, dict):
                raise ValueError("pop_dict must be a dict mapping item_id -> probability")
            pairs = []
            for k, v in pop_dict.items():
                kk = int(k)
                if 0 <= kk < int(item_num):
                    pairs.append((kk, float(v)))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            k = int(min(int(eval_neg_samples_num), int(item_num)))
            top_ids = [kk + 1 for kk, _ in pairs[:k]]
            sampled_negatives = torch.as_tensor(top_ids, device=device, dtype=torch.long)

    if use_bert4rec_loo and model_type != "albert4rec":

        def eval_fn(model, session_loader, reward_click, reward_buy, device, **kwargs):
            return evaluate_loo_candidates(
                model,
                session_loader,
                reward_click,
                reward_buy,
                device,
                sampled_negatives=sampled_negatives,
                **kwargs,
            )

    else:
        eval_fn = evaluate_loo if use_bert4rec_loo else evaluate

    reward_click = float(cfg.get("r_click", 0.2))
    reward_buy = float(cfg.get("r_buy", 1.0))
    rneg_cfg = cfg.get("r_negative", -0.0)
    if isinstance(rneg_cfg, str) and ("(" in rneg_cfg) and rneg_cfg.strip().endswith(")"):
        gs_cfg0 = cfg.get("gridsearch") or {}
        if bool(gs_cfg0.get("enable", False)):
            reward_negative = 0.0
        else:
            reward_negative = float(rneg_cfg)
    else:
        reward_negative = float(rneg_cfg)
    purchase_only = bool(cfg.get("purchase_only", False))

    if not smoke_cpu:
        train_batch_size = int(cfg.get("batch_size_train", 256))
        val_batch_size = int(cfg.get("batch_size_val", 256))
        train_num_workers = int(cfg.get("num_workers_train", 0))
        val_num_workers = int(cfg.get("num_workers_val", 0))

    pin_memory = True

    if persrec_tc5:
        train_ds_s = 0.0
        val_ds_s = 0.0
        test_ds_s = 0.0
    else:
        if use_bert4rec_loo:
            train_ds, val_ds, test_ds = prepare_sessions_bert4rec_loo(
                data_directory=data_directory,
                split_df_names=["sampled_train.df", "sampled_val.df", "sampled_test.df"],
                seed=int(cfg.get("seed", 0)),
                val_samples_num=int(val_split_samples_num),
                test_samples_num=int(test_split_samples_num),
                limit_chunks_pct=limit_chunks_pct,
            )
            train_ds_s = 0.0
            val_ds_s = 0.0
            test_ds_s = 0.0
        else:
            t0 = time.perf_counter()
            train_ds = SessionDataset(data_directory=data_directory, split_df_name="sampled_train.df")
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

    if (not persrec_tc5) and (not use_bert4rec_loo):
        t0 = time.perf_counter()
        val_ds = SessionDataset(data_directory=data_directory, split_df_name="sampled_val.df")
        val_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if model_type == "albert4rec":
        val_dl = make_albert4rec_loader(
            val_ds,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            state_size=int(state_size),
            purchase_only=bool(purchase_only),
            shuffle=False,
        )
    else:
        val_dl = make_session_loader(
            val_ds,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            pad_item=item_num,
            shuffle=False,
        )
    val_dl_s = time.perf_counter() - t0

    if (not persrec_tc5) and (not use_bert4rec_loo):
        t0 = time.perf_counter()
        test_ds = SessionDataset(data_directory=data_directory, split_df_name="sampled_test.df")
        test_ds_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if model_type == "albert4rec":
        test_dl = make_albert4rec_loader(
            test_ds,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            state_size=int(state_size),
            purchase_only=bool(purchase_only),
            shuffle=False,
        )
    else:
        test_dl = make_session_loader(
            test_ds,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            pad_item=item_num,
            shuffle=False,
        )
    test_dl_s = time.perf_counter() - t0

    gs_cfg = cfg.get("gridsearch") or {}
    if is_distributed() and bool(gs_cfg.get("enable", False)):
        raise ValueError("gridsearch.enable=true is not supported with DDP")
    if model_type == "albert4rec" and bool(gs_cfg.get("enable", False)):
        raise ValueError("gridsearch is not supported for albert4rec")
    if trainer == "crr" and bool(gs_cfg.get("enable", False)):
        raise ValueError("gridsearch is not supported for trainer=crr")
    if continue_training and bool(gs_cfg.get("enable", False)):
        raise ValueError("--continue is not supported with gridsearch.enable=true")
    if eval_only:
        if is_distributed() and (not is_rank0()):
            barrier()
            return
        ckpt_run_dir = run_dir
        if persrec_tc5 and eval_scheme in {"bert4rec_eval_plu-disable", "bert4rec_eval_plu-inverse"}:
            ckpt_run_dir = make_run_dir(dataset_name, config_name, eval_scheme="bert4rec_eval")
            for fname in ("best_model.pt", "best_model_warmup.pt", "config.yml"):
                src = ckpt_run_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(run_dir / fname))

        best_path = ckpt_run_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {best_path}")

        a4_cfg = cfg.get("albert4rec") or {}
        intermediate_size = a4_cfg.get("intermediate_size", None) if isinstance(a4_cfg, dict) else None
        if intermediate_size is not None:
            intermediate_size = int(intermediate_size)

        if model_type == "albert4rec":
            best_model = Albert4Rec(
                item_num=item_num,
                state_size=state_size,
                hidden_size=int(cfg.get("hidden_factor", 64)),
                num_heads=int(cfg.get("num_heads", 1)),
                num_layers=int(cfg.get("num_blocks", 1)),
                dropout_rate=float(cfg.get("dropout_rate", 0.1)),
                intermediate_size=intermediate_size,
            ).to(device)
            eval_fn_eff = evaluate_albert4rec_loo
        else:
            if enable_sa2c:
                best_model = SASRecQNetworkRectools(
                    item_num=item_num,
                    state_size=state_size,
                    hidden_size=int(cfg.get("hidden_factor", 64)),
                    num_heads=int(cfg.get("num_heads", 1)),
                    num_blocks=int(cfg.get("num_blocks", 1)),
                    dropout_rate=float(cfg.get("dropout_rate", 0.1)),
                    pointwise_critic_use=pointwise_critic_use,
                    pointwise_critic_arch=pointwise_critic_arch,
                    pointwise_critic_mlp=pointwise_mlp_cfg,
                    actor_lstm=actor_lstm_cfg,
                    actor_mlp=actor_mlp_cfg,
                    critic_lstm=critic_lstm_cfg,
                    critic_mlp=critic_mlp_cfg,
                ).to(device)
            else:
                best_model = SASRecBaselineRectools(
                    item_num=item_num,
                    state_size=state_size,
                    hidden_size=int(cfg.get("hidden_factor", 64)),
                    num_heads=int(cfg.get("num_heads", 1)),
                    num_blocks=int(cfg.get("num_blocks", 1)),
                    dropout_rate=float(cfg.get("dropout_rate", 0.1)),
                ).to(device)
            eval_fn_eff = eval_fn
        best_model.load_state_dict(torch.load(best_path, map_location=device))

        val_best = eval_fn_eff(
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
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )
        test_best = eval_fn_eff(
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
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )

        val_warmup = None
        test_warmup = None
        if enable_sa2c and model_type != "albert4rec":
            warmup_path = ckpt_run_dir / "best_model_warmup.pt"
            if not warmup_path.exists():
                warmup_path = ckpt_run_dir / "best_warmup_model.pt"
            if warmup_path.exists():
                warmup_model = SASRecQNetworkRectools(
                    item_num=item_num,
                    state_size=state_size,
                    hidden_size=int(cfg.get("hidden_factor", 64)),
                    num_heads=int(cfg.get("num_heads", 1)),
                    num_blocks=int(cfg.get("num_blocks", 1)),
                    dropout_rate=float(cfg.get("dropout_rate", 0.1)),
                    pointwise_critic_use=pointwise_critic_use,
                    pointwise_critic_arch=pointwise_critic_arch,
                    pointwise_critic_mlp=pointwise_mlp_cfg,
                    actor_lstm=actor_lstm_cfg,
                    actor_mlp=actor_mlp_cfg,
                    critic_lstm=critic_lstm_cfg,
                    critic_mlp=critic_mlp_cfg,
                ).to(device)
                warmup_model.load_state_dict(torch.load(warmup_path, map_location=device))
                val_warmup = eval_fn(
                    warmup_model,
                    val_dl,
                    reward_click,
                    reward_buy,
                    device,
                    debug=bool(cfg.get("debug", False)),
                    split="val(best_warmup)",
                    state_size=state_size,
                    item_num=item_num,
                    purchase_only=purchase_only,
                    ce_loss_vocab_size=ce_loss_vocab_size,
                    ce_full_vocab_size=ce_full_vocab_size,
                    ce_vocab_pct=ce_vocab_pct,
                )
                test_warmup = eval_fn(
                    warmup_model,
                    test_dl,
                    reward_click,
                    reward_buy,
                    device,
                    debug=bool(cfg.get("debug", False)),
                    split="test(best_warmup)",
                    state_size=state_size,
                    item_num=item_num,
                    purchase_only=purchase_only,
                    ce_loss_vocab_size=ce_loss_vocab_size,
                    ce_full_vocab_size=ce_full_vocab_size,
                    ce_vocab_pct=ce_vocab_pct,
                )

        if is_rank0():
            write_results(
                run_dir=run_dir,
                val_best=val_best,
                test_best=test_best,
                val_warmup=val_warmup,
                test_warmup=test_warmup,
                smoke_cpu=smoke_cpu,
            )
        if is_distributed():
            barrier()
        return

    if bool(gs_cfg.get("enable", False)):
        run_optuna_gridsearch(
            cfg=cfg,
            base_run_dir=run_dir,
            device=device,
            train_ds=train_ds,
            val_dl=val_dl,
            test_dl=test_dl,
            pop_dict_path=Path(pop_dict_path) if enable_sa2c else None,
            reward_click=reward_click,
            reward_buy=reward_buy,
            reward_negative=reward_negative,
            state_size=state_size,
            item_num=item_num,
            purchase_only=purchase_only,
            num_batches=num_batches,
            train_batch_size=train_batch_size,
            train_num_workers=train_num_workers,
            pin_memory=pin_memory,
            reward_fn=reward_fn,
            smoke_cpu=smoke_cpu,
        )
        return

    if model_type == "albert4rec":
        a4_cfg = cfg.get("albert4rec") or {}
        intermediate_size = a4_cfg.get("intermediate_size", None) if isinstance(a4_cfg, dict) else None
        if intermediate_size is not None:
            intermediate_size = int(intermediate_size)
        best_path = train_albert4rec(
            cfg=cfg,
            train_ds=train_ds,
            val_dl=val_dl,
            run_dir=run_dir,
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
        )
        if is_distributed():
            barrier()
        warmup_path = None
        best_model = Albert4Rec(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_layers=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            intermediate_size=intermediate_size,
        ).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        eval_fn_eff = evaluate_albert4rec_loo
    elif trainer == "crr":
        best_path = train_crr(
            cfg=cfg,
            train_ds=train_ds,
            val_dl=val_dl,
            run_dir=run_dir,
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
            reward_fn=reward_fn,
            evaluate_fn=eval_fn,
        )
        if is_distributed():
            barrier()
        warmup_path = None
        best_model = SASRecQNetworkRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            pointwise_critic_use=pointwise_critic_use,
            pointwise_critic_arch=pointwise_critic_arch,
            pointwise_critic_mlp=pointwise_mlp_cfg,
            actor_lstm=actor_lstm_cfg,
            actor_mlp=actor_mlp_cfg,
            critic_lstm=critic_lstm_cfg,
            critic_mlp=critic_mlp_cfg,
        ).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
    elif trainer == "sa2c":
        best_path, warmup_path = train_sa2c(
            cfg=cfg,
            train_ds=train_ds,
            val_dl=val_dl,
            pop_dict_path=Path(pop_dict_path),
            run_dir=run_dir,
            device=device,
            reward_click=reward_click,
            reward_buy=reward_buy,
            reward_negative=reward_negative,
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
            evaluate_fn=eval_fn,
            continue_training=continue_training,
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )
        if is_distributed():
            barrier()
        best_model = SASRecQNetworkRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            pointwise_critic_use=pointwise_critic_use,
            pointwise_critic_arch=pointwise_critic_arch,
            pointwise_critic_mlp=pointwise_mlp_cfg,
        ).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        best_path = train_baseline(
            cfg=cfg,
            train_ds=train_ds,
            val_dl=val_dl,
            run_dir=run_dir,
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
            evaluate_fn=eval_fn,
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )
        if is_distributed():
            barrier()
        warmup_path = None
        best_model = SASRecBaselineRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        ).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        eval_fn_eff = eval_fn

    if model_type != "albert4rec":
        eval_fn_eff = eval_fn

    if is_distributed() and (not is_rank0()):
        barrier()
        return

    val_best = eval_fn_eff(
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
        ce_loss_vocab_size=ce_loss_vocab_size,
        ce_full_vocab_size=ce_full_vocab_size,
        ce_vocab_pct=ce_vocab_pct,
    )
    test_best = eval_fn_eff(
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
        ce_loss_vocab_size=ce_loss_vocab_size,
        ce_full_vocab_size=ce_full_vocab_size,
        ce_vocab_pct=ce_vocab_pct,
    )

    val_warmup = None
    test_warmup = None
    if warmup_path is not None and Path(warmup_path).exists() and model_type != "albert4rec":
        warmup_model = SASRecQNetworkRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            pointwise_critic_use=pointwise_critic_use,
            pointwise_critic_arch=pointwise_critic_arch,
            pointwise_critic_mlp=pointwise_mlp_cfg,
            actor_lstm=actor_lstm_cfg,
            actor_mlp=actor_mlp_cfg,
            critic_lstm=critic_lstm_cfg,
            critic_mlp=critic_mlp_cfg,
        ).to(device)
        warmup_model.load_state_dict(torch.load(warmup_path, map_location=device))

        val_warmup = eval_fn(
            warmup_model,
            val_dl,
            reward_click,
            reward_buy,
            device,
            debug=bool(cfg.get("debug", False)),
            split="val(best_warmup)",
            state_size=state_size,
            item_num=item_num,
            purchase_only=purchase_only,
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )
        test_warmup = eval_fn(
            warmup_model,
            test_dl,
            reward_click,
            reward_buy,
            device,
            debug=bool(cfg.get("debug", False)),
            split="test(best_warmup)",
            state_size=state_size,
            item_num=item_num,
            purchase_only=purchase_only,
            ce_loss_vocab_size=ce_loss_vocab_size,
            ce_full_vocab_size=ce_full_vocab_size,
            ce_vocab_pct=ce_vocab_pct,
        )

    if is_rank0():
        write_results(
            run_dir=run_dir,
            val_best=val_best,
            test_best=test_best,
            val_warmup=val_warmup,
            test_warmup=test_warmup,
            smoke_cpu=smoke_cpu,
        )
    if is_distributed():
        barrier()


def _spawn_entry(
    local_rank: int,
    world_size: int,
    device_ids: list[int],
    cfg: dict,
    args,
) -> None:
    silence_logging_if_needed(is_rank0=(int(local_rank) == 0))
    os.environ["RANK"] = str(int(local_rank))
    os.environ["LOCAL_RANK"] = str(int(local_rank))
    os.environ["WORLD_SIZE"] = str(int(world_size))
    device_idx = int(device_ids[int(local_rank)])
    torch.cuda.set_device(int(device_idx))
    ddp_setup(world_size=int(world_size))
    try:
        _worker_main(cfg=cfg, args=args, local_rank=int(local_rank), world_size=int(world_size), device_ids=device_ids)
    finally:
        ddp_cleanup()


def main():
    args = parse_args()
    config_path = args.config
    cfg = load_config(config_path)
    cfg = apply_cli_overrides(cfg, args)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size_env > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
        silence_logging_if_needed(is_rank0=(int(local_rank) == 0))
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 but CUDA is not available")
        n_visible = int(torch.cuda.device_count())
        if local_rank < 0 or local_rank >= n_visible:
            raise RuntimeError(f"Invalid LOCAL_RANK={local_rank} for visible cuda device_count={n_visible}")
        torch.cuda.set_device(int(local_rank))
        ddp_setup(world_size=int(world_size_env))
        try:
            _worker_main(cfg=cfg, args=args, local_rank=int(local_rank), world_size=int(world_size_env), device_ids=None)
        finally:
            ddp_cleanup()
        return

    device_ids = parse_cuda_devices(cfg.get("device_id", None))
    if len(device_ids) <= 1:
        if len(device_ids) == 1 and torch.cuda.is_available():
            torch.cuda.set_device(int(device_ids[0]))
        _worker_main(cfg=cfg, args=args, local_rank=0, world_size=1, device_ids=device_ids if device_ids else None)
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    world_size = int(len(device_ids))
    mp.spawn(
        _spawn_entry,
        args=(world_size, device_ids, cfg, args),
        nprocs=world_size,
        join=True,
    )


__all__ = ["main"]

