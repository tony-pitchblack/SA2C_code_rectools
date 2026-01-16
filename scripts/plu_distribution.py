from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm.auto import tqdm


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * (float(part) / float(total))


def _iter_token_arrays(table: pa.Table, product_column: str) -> list[pa.Array]:
    col = table[product_column]
    out: list[pa.Array] = []
    for chunk in col.chunks:
        if pa.types.is_list(chunk.type) or pa.types.is_large_list(chunk.type):
            out.append(chunk)
            continue
        if pa.types.is_string(chunk.type) or pa.types.is_large_string(chunk.type):
            seqs = chunk.to_pylist()
            lists: list[list[int]] = []
            for s in seqs:
                if s is None:
                    lists.append([])
                    continue
                ss = str(s).strip()
                if not ss:
                    lists.append([])
                    continue
                parts = [p.strip() for p in ss.split(",") if p.strip()]
                if not parts:
                    lists.append([])
                    continue
                lists.append([int(x) for x in parts])
            out.append(pa.array(lists, type=pa.list_(pa.int64())))
            continue
        raise TypeError(f"Unsupported {product_column} type: {chunk.type}")
    return out


def _process_list_array(arr: pa.Array):
    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
        raise TypeError(f"Expected list array, got {arr.type}")
    values = arr.values
    if pa.types.is_integer(values.type):
        values_np = values.to_numpy(zero_copy_only=False)
    else:
        values_np = values.cast(pa.int64()).to_numpy(zero_copy_only=False)
    values_np = values_np.astype(np.int64, copy=False)

    is_plu = values_np >= 0
    total_tokens = int(values_np.size)
    plu_tokens = int(is_plu.sum())
    non_plu_tokens = int(total_tokens - plu_tokens)

    if total_tokens > 0:
        uniq = np.unique(values_np)
        uniq_plu = int((uniq >= 0).sum())
        uniq_non_plu = int(uniq.size - uniq_plu)
    else:
        uniq = np.empty((0,), dtype=np.int64)
        uniq_plu = 0
        uniq_non_plu = 0

    offsets_np = arr.offsets.to_numpy(zero_copy_only=False)
    offsets_np = offsets_np.astype(np.int64, copy=False)
    if int(offsets_np.size) == 0:
        lengths = np.empty((0,), dtype=np.int64)
        plu_per_row = np.empty((0,), dtype=np.int64)
    else:
        lengths = np.diff(offsets_np).astype(np.int64, copy=False)
        if int(values_np.size) == 0:
            plu_per_row = np.zeros((int(lengths.size),), dtype=np.int64)
        else:
            start = offsets_np[:-1]
            plu_per_row = np.add.reduceat(is_plu.astype(np.int64, copy=False), start)
            if int(plu_per_row.size) != int(lengths.size):
                raise RuntimeError("Unexpected reduceat output shape")

    return {
        "unique_tokens": uniq,
        "unique_plu": uniq_plu,
        "unique_non_plu": uniq_non_plu,
        "total_tokens": total_tokens,
        "plu_tokens": plu_tokens,
        "non_plu_tokens": non_plu_tokens,
        "seq_lengths": lengths,
        "seq_plu_counts": plu_per_row,
    }


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

    dataset = ds.dataset(str(parquet_dir), format="parquet")
    product_column = str(args.product_column)
    if product_column not in dataset.schema.names:
        raise KeyError(f"Missing column {product_column!r} in parquet schema: {dataset.schema.names}")

    logger.info("parquet_dir=%s", str(parquet_dir))
    logger.info("product_column=%s", str(product_column))
    try:
        logger.info("parquet_schema=%s", dataset.schema)
    except Exception:
        pass

    unique_tokens: set[int] = set()

    total_tokens = 0
    total_plu_tokens = 0
    total_non_plu_tokens = 0

    seq_count = 0
    seq_plu_frac_sum = 0.0
    seq_non_plu_frac_sum = 0.0

    scanner = dataset.scanner(columns=[product_column], batch_size=1024)
    logger.info("scan_start")
    it = scanner.to_batches()
    if not bool(args.no_tqdm):
        it = tqdm(it, desc="scan parquet", unit="batch", dynamic_ncols=True)
    for batch in it:
        table = pa.Table.from_batches([batch])
        arrays = _iter_token_arrays(table, product_column)
        for arr in arrays:
            stats = _process_list_array(arr)

            uniq = stats["unique_tokens"]
            if int(uniq.size) > 0:
                for x in uniq.tolist():
                    unique_tokens.add(int(x))

            total_tokens += int(stats["total_tokens"])
            total_plu_tokens += int(stats["plu_tokens"])
            total_non_plu_tokens += int(stats["non_plu_tokens"])

            lengths = stats["seq_lengths"]
            plu_counts = stats["seq_plu_counts"]
            if int(lengths.size) > 0:
                nonzero = lengths > 0
                if bool(nonzero.any()):
                    L = lengths[nonzero].astype(np.float64, copy=False)
                    P = plu_counts[nonzero].astype(np.float64, copy=False)
                    frac_plu = P / L
                    seq_plu_frac_sum += float(frac_plu.sum())
                    seq_non_plu_frac_sum += float((1.0 - frac_plu).sum())
                    seq_count += int(frac_plu.size)

        if not bool(args.no_tqdm) and hasattr(it, "set_postfix"):
            try:
                it.set_postfix(
                    tokens=int(total_tokens),
                    unique=int(len(unique_tokens)),
                    seqs=int(seq_count),
                )
            except Exception:
                pass

    logger.info("scan_done tokens=%d unique=%d seqs=%d", int(total_tokens), int(len(unique_tokens)), int(seq_count))

    uniq_total = int(len(unique_tokens))
    uniq_plu_total = int(sum(1 for x in unique_tokens if int(x) >= 0))
    uniq_non_plu_total = int(uniq_total - uniq_plu_total)

    print(f"unique_tokens_total={uniq_total}")
    print(f"unique_plu={uniq_plu_total} ({_pct(uniq_plu_total, uniq_total):.6f}%)")
    print(f"unique_non_plu={uniq_non_plu_total} ({_pct(uniq_non_plu_total, uniq_total):.6f}%)")
    print("")
    print(f"all_tokens_total={int(total_tokens)}")
    print(f"all_tokens_plu={int(total_plu_tokens)} ({_pct(int(total_plu_tokens), int(total_tokens)):.6f}%)")
    print(f"all_tokens_non_plu={int(total_non_plu_tokens)} ({_pct(int(total_non_plu_tokens), int(total_tokens)):.6f}%)")
    print("")
    if seq_count <= 0:
        avg_plu = 0.0
        avg_non_plu = 0.0
    else:
        avg_plu = 100.0 * (seq_plu_frac_sum / float(seq_count))
        avg_non_plu = 100.0 * (seq_non_plu_frac_sum / float(seq_count))
    if not math.isfinite(avg_plu):
        avg_plu = 0.0
    if not math.isfinite(avg_non_plu):
        avg_non_plu = 0.0
    print(f"avg_seq_plu_pct={avg_plu:.6f}%")
    print(f"avg_seq_non_plu_pct={avg_non_plu:.6f}%")


if __name__ == "__main__":
    main()

