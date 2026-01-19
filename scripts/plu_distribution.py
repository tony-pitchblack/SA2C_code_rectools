from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


def _is_plu_id(x: int) -> bool:
    return int(x) >= 0


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * (float(part) / float(total))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-working-prefix", required=True)
    p.add_argument("--product-column", default="product_id")
    p.add_argument("--log-level", default="INFO")
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
    logger.info("scan_start")
    dataset = ds.dataset(str(parquet_dir), format="parquet")
    try:
        sch = dataset.schema
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet schema from {str(parquet_dir)}") from e
    if product_column not in set(sch.names):
        raise KeyError(f"Missing column {product_column!r} in parquet schema columns: {list(sch.names)}")
    col_type = sch.field(product_column).type

    print(f"column_dtype={col_type}")
    first_example = None
    try:
        scanner0 = dataset.scanner(columns=[product_column], batch_size=1024)
        for b0 in scanner0.to_batches():
            c0 = b0.column(product_column)
            if c0.null_count:
                c0 = pc.fill_null(c0, pa.scalar([], type=c0.type)) if (pa.types.is_list(col_type) or pa.types.is_large_list(col_type)) else c0
            if len(c0) == 0:
                continue
            if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                for v in c0.to_pylist():
                    if v is not None:
                        first_example = v
                        break
            else:
                first_example = c0[0].as_py()
            if first_example is not None:
                break
    except Exception:
        first_example = None
    print(f"example_sequence={first_example}")

    total_tokens = 0
    total_plu_tokens = 0
    uniq_set: set[int] = set()
    sum_frac_plu = 0.0
    n_seqs = 0

    scanner = dataset.scanner(columns=[product_column])
    if not (pa.types.is_list(col_type) or pa.types.is_large_list(col_type)):
        raise TypeError(
            f"Expected {product_column!r} to be a list column, got {col_type}. "
            "Expected list<int> or list<string> where elements are numeric strings."
        )

    for batch in scanner.to_batches():
        col = batch.column(product_column)
        if col.null_count:
            col = pc.fill_null(col, pa.scalar([], type=col.type))

        lengths = pc.list_value_length(col)
        lengths_np = np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)
        if lengths_np.size == 0:
            continue

        total_tokens += int(lengths_np.sum())

        flat = pc.list_flatten(col)
        if len(flat) > 0:
            flat_i64 = pc.cast(flat, pa.int64(), safe=False)
            flat_np = np.asarray(flat_i64.to_numpy(zero_copy_only=False), dtype=np.int64)
            is_plu_flat = flat_np >= 0
            total_plu_tokens += int(is_plu_flat.sum())

            uniq_batch = pc.unique(flat_i64)
            uniq_set.update(int(x.as_py()) for x in uniq_batch)

            offsets = np.asarray(col.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
            plu_prefix = np.concatenate(([0], np.cumsum(is_plu_flat.astype(np.int64, copy=False), dtype=np.int64)))
            plu_counts = plu_prefix[offsets[1:]] - plu_prefix[offsets[:-1]]
        else:
            plu_counts = np.zeros_like(lengths_np, dtype=np.int64)

        nonempty = lengths_np > 0
        if bool(nonempty.any()):
            frac = plu_counts[nonempty].astype(np.float64, copy=False) / lengths_np[nonempty].astype(np.float64, copy=False)
            sum_frac_plu += float(frac.sum())
            n_seqs += int(nonempty.sum())

    logger.info("scan_done")

    total_non_plu_tokens = int(total_tokens - total_plu_tokens)
    uniq_total = int(len(uniq_set))
    uniq_plu_total = int(sum(1 for x in uniq_set if _is_plu_id(x)))
    uniq_non_plu_total = int(uniq_total - uniq_plu_total)
    avg_plu = float(100.0 * (sum_frac_plu / float(n_seqs))) if n_seqs > 0 else 0.0
    avg_non_plu = float(100.0 - avg_plu) if np.isfinite(avg_plu) else 0.0

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

