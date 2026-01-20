import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train SA2C (Rectools) with per-position SASRec logits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--continue",
        dest="continue_training",
        type=str,
        default=None,
        metavar="MLFLOW_RUN_ID",
        help="Continue training from run_dir checkpoints and log to an existing MLflow run_id.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing best checkpoints in the corresponding run_dir and write results.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Enable persrec_tc5 sanity subset loading and use *_sanity run dir suffix.",
    )
    parser.add_argument("--early_stopping_ep", type=int, default=None, help="Patience epochs for early stopping.")
    parser.add_argument("--early_stopping_metric", type=str, default=None, help="Early stopping metric (ndcg@10).")
    parser.add_argument("--max_steps", type=int, default=None, help="If set, stop after this many update steps.")
    parser.add_argument(
        "--batch-size-pct",
        dest="batch_size_pct",
        type=float,
        default=None,
        help="Scale batch_size_train and batch_size_val from config by this factor (>0).",
    )
    parser.add_argument(
        "--plu-filter",
        dest="plu_filter",
        choices=("enable", "disable", "inverse"),
        default=argparse.SUPPRESS,
        help="persrec_tc5 only: enable (default), disable, inverse.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging and NaN checks (overrides config).")
    parser.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Force CPU, set batch_size=8, run 1 epoch, and skip writing val/test result files (keeps logging).",
    )
    return parser.parse_args()


__all__ = ["parse_args"]

