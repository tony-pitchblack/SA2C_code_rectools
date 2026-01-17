from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm.auto import tqdm


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * (float(part) / float(total))


def _normalize_sequences(series: pd.Series) -> pd.Series:
    def _to_list(x) -> list[int]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, (list, tuple, np.ndarray)):
            return [int(v) for v in list(x)]
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return [int(v) for v in parts]
        try:
            return [int(x)]
        except Exception as e:
            raise TypeError(f"Unsupported sequence cell type: {type(x)}") from e

    return series.map(_to_list)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-working-prefix", required=True)
    p.add_argument("--product-column", default="product_id")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--no-tqdm", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=str(args.log_level).upper(), format="[%(asctime)s] %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    root = Path(str(args.local_working_prefix)).expanduser().resolve()
    parquet_dir = root / "dataset_train.parquet"
    if not parquet_dir.exists():
        raise FileNotFoundError(str(parquet_dir))

    product_column = str(args.product_column)
    logger.info("parquet_dir=%s", str(parquet_dir))
    logger.info("product_column=%s", str(product_column))
    logger.info("load_start")
    df = pd.read_parquet(str(parquet_dir), columns=[product_column])
    if product_column not in df.columns:
        raise KeyError(f"Missing column {product_column!r} in parquet dataframe columns: {list(df.columns)}")
    logger.info("load_done rows=%d", int(len(df)))

    logger.info("normalize_sequences_start")
    seqs = _normalize_sequences(df[product_column])
    logger.info("normalize_sequences_done")

    logger.info("explode_start")
    ex = seqs.explode(ignore_index=False)
    ex = ex.dropna()
    ex = pd.to_numeric(ex, errors="raise").astype("int64", copy=False)
    logger.info("explode_done")

    logger.info("count_tokens_start")
    total_tokens = int(ex.shape[0])
    is_plu = ex.ge(0)
    total_plu_tokens = int(is_plu.sum())
    total_non_plu_tokens = int(total_tokens - total_plu_tokens)

    uniq = ex.unique()
    uniq_total = int(uniq.size)
    uniq_plu_total = int((uniq >= 0).sum())
    uniq_non_plu_total = int(uniq_total - uniq_plu_total)
    logger.info("count_tokens_done total_tokens=%d unique_tokens=%d", int(total_tokens), int(uniq_total))

    logger.info("per_sequence_start")
    per_seq_total = ex.groupby(level=0, sort=False).size().astype("int64", copy=False)
    per_seq_plu = is_plu.groupby(level=0, sort=False).sum().astype("int64", copy=False)
    frac_plu = (per_seq_plu / per_seq_total).astype("float64", copy=False)
    avg_plu = float(100.0 * frac_plu.mean()) if int(frac_plu.shape[0]) > 0 else 0.0
    avg_non_plu = float(100.0 - avg_plu) if np.isfinite(avg_plu) else 0.0
    logger.info("per_sequence_done seqs=%d", int(frac_plu.shape[0]))

    print(f"unique_tokens_total={uniq_total}")
    print(f"unique_plu={uniq_plu_total} ({_pct(uniq_plu_total, uniq_total):.6f}%)")
    print(f"unique_non_plu={uniq_non_plu_total} ({_pct(uniq_non_plu_total, uniq_total):.6f}%)")
    print("")
    print(f"all_tokens_total={int(total_tokens)}")
    print(f"all_tokens_plu={int(total_plu_tokens)} ({_pct(int(total_plu_tokens), int(total_tokens)):.6f}%)")
    print(f"all_tokens_non_plu={int(total_non_plu_tokens)} ({_pct(int(total_non_plu_tokens), int(total_tokens)):.6f}%)")
    print("")
    print(f"avg_seq_plu_pct={avg_plu:.6f}%")
    print(f"avg_seq_non_plu_pct={avg_non_plu:.6f}%")


if __name__ == "__main__":
    main()

