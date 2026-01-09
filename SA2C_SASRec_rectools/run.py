from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import write_results
from .cli import parse_args
from .config import apply_cli_overrides, is_persrec_tc5_dataset_cfg, load_config
from .data.bert4rec_loo import prepare_persrec_tc5_bert4rec_loo, prepare_sessions_bert4rec_loo
from .data.persrec_tc5 import prepare_persrec_tc5
from .data.sessions import SessionDataset, make_session_loader
from .logging_utils import configure_logging, dump_config
from .models import SASRecBaselineRectools, SASRecQNetworkRectools
from .paths import make_run_dir, resolve_dataset_root
from .metrics import evaluate, evaluate_loo
from .training.baseline import train_baseline
from .training.sa2c import train_sa2c


def main():
    args = parse_args()
    config_path = args.config
    cfg = load_config(config_path)
    cfg = apply_cli_overrides(cfg, args)

    if str(cfg.get("early_stopping_metric", "ndcg@10")) != "ndcg@10":
        raise ValueError("Only early_stopping_metric='ndcg@10' is supported.")
    reward_fn = str(cfg.get("reward_fn", "click_buy"))
    if reward_fn not in {"click_buy", "ndcg"}:
        raise ValueError("reward_fn must be one of: click_buy | ndcg")
    enable_sa2c = bool(cfg.get("enable_sa2c", True))

    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = cfg.get("dataset", "retailrocket")
    persrec_tc5 = is_persrec_tc5_dataset_cfg(dataset_cfg)
    if persrec_tc5:
        calc_date = str(dataset_cfg.get("calc_date"))
        dataset_name = f"persrec_tc5_{calc_date}"
        dataset_root = repo_root
    else:
        dataset_name = str(dataset_cfg)
        dataset_root = resolve_dataset_root(dataset_name)

    config_name = Path(config_path).stem
    if bool(getattr(args, "sanity", False)):
        config_name = f"{config_name}_sanity"
    run_dir = make_run_dir(dataset_name, config_name)
    configure_logging(run_dir, debug=bool(cfg.get("debug", False)))
    dump_config(cfg, run_dir)

    logger = logging.getLogger(__name__)
    logger.info("run_dir: %s", str(run_dir))
    logger.info("dataset: %s", dataset_name)
    if bool(getattr(args, "smoke_cpu", False)):
        logger.info("smoke_cpu: enabled (forcing CPU, batch_size=8, epoch=1, skipping val/test result file writing)")

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bert4rec_loo_cfg = cfg.get("bert4rec_loo") or {}
    use_bert4rec_loo = bool(isinstance(bert4rec_loo_cfg, dict) and bool(bert4rec_loo_cfg.get("enable", False)))
    eval_fn = evaluate_loo if use_bert4rec_loo else evaluate

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
    if persrec_tc5:
        if use_bert4rec_loo:
            data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds = prepare_persrec_tc5_bert4rec_loo(
                dataset_root=dataset_root,
                data_rel=data_rel,
                dataset_name=dataset_name,
                dataset_cfg=dict(dataset_cfg),
                seed=int(cfg.get("seed", 0)),
                val_samples_num=int(bert4rec_loo_cfg.get("val_samples_num", 0)),
                test_samples_num=int(bert4rec_loo_cfg.get("test_samples_num", 0)),
            )
        else:
            data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds = prepare_persrec_tc5(
                dataset_root=dataset_root,
                data_rel=data_rel,
                dataset_name=dataset_name,
                dataset_cfg=dict(dataset_cfg),
                seed=int(cfg.get("seed", 0)),
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

    reward_click = float(cfg.get("r_click", 0.2))
    reward_buy = float(cfg.get("r_buy", 1.0))
    reward_negative = float(cfg.get("r_negative", -0.0))
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
                val_samples_num=int(bert4rec_loo_cfg.get("val_samples_num", 0)),
                test_samples_num=int(bert4rec_loo_cfg.get("test_samples_num", 0)),
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
    test_dl = make_session_loader(
        test_ds,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        pad_item=item_num,
        shuffle=False,
    )
    test_dl_s = time.perf_counter() - t0

    if enable_sa2c:
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
        )
        best_model = SASRecQNetworkRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
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
        )
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

    val_best = eval_fn(
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
    test_best = eval_fn(
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

    val_warmup = None
    test_warmup = None
    if warmup_path is not None and Path(warmup_path).exists():
        warmup_model = SASRecQNetworkRectools(
            item_num=item_num,
            state_size=state_size,
            hidden_size=int(cfg.get("hidden_factor", 64)),
            num_heads=int(cfg.get("num_heads", 1)),
            num_blocks=int(cfg.get("num_blocks", 1)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
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
        )

    write_results(
        run_dir=run_dir,
        val_best=val_best,
        test_best=test_best,
        val_warmup=val_warmup,
        test_warmup=test_warmup,
        smoke_cpu=smoke_cpu,
    )


__all__ = ["main"]

