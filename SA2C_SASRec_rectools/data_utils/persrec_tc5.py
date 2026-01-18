from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import tqdm


def hdfs_get(src: str, dst: str):
    logger = logging.getLogger(__name__)
    candidates = []
    if shutil.which("hdfs") is not None:
        candidates.append(["hdfs", "dfs", "-get", src, dst])
    if shutil.which("hadoop") is not None:
        candidates.append(["hadoop", "fs", "-get", src, dst])
    if not candidates:
        raise RuntimeError("Neither 'hdfs' nor 'hadoop' was found in PATH; cannot download dataset from HDFS.")
    last_err = None
    for cmd in candidates:
        try:
            logger.info("persrec_tc5: hdfs_get using `%s` (%s -> %s)", " ".join(cmd[:2]), str(src), str(dst))
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            out = (e.stdout or "").strip()
            if out:
                logger.warning("HDFS download failed with %s: %s", " ".join(cmd[:2]), out.splitlines()[-1])
            continue
    raise RuntimeError(f"Failed to download dataset from HDFS. Last error: {last_err}") from last_err


def ensure_local_parquet_cache(*, hdfs_working_prefix: str, local_parquet_dir: Path):
    logger = logging.getLogger(__name__)
    local_parquet_dir = Path(local_parquet_dir)
    if local_parquet_dir.exists() and any(local_parquet_dir.iterdir()):
        logger.info("persrec_tc5: using raw parquet cache at %s", str(local_parquet_dir))
        return
    local_parquet_dir.parent.mkdir(parents=True, exist_ok=True)
    src = str(Path(hdfs_working_prefix) / "training" / "dataset_train.parquet")
    dst = str(local_parquet_dir)
    t0 = time.perf_counter()
    logger.info("persrec_tc5: raw parquet cache missing at %s -> downloading from %s", str(local_parquet_dir), str(src))
    hdfs_get(src, dst)
    logger.info("persrec_tc5: hdfs download done in %.3fs", float(time.perf_counter() - t0))
    if not local_parquet_dir.exists():
        raise RuntimeError(f"HDFS download completed but local path does not exist: {str(local_parquet_dir)}")


def _list_parquet_part_files(parquet_dir: Path) -> list[Path]:
    parquet_dir = Path(parquet_dir)
    if parquet_dir.is_file() and parquet_dir.suffix == ".parquet":
        return [parquet_dir]
    if parquet_dir.is_dir():
        return sorted([p for p in parquet_dir.iterdir() if p.is_file() and p.suffix == ".parquet"], key=lambda p: p.name)
    raise FileNotFoundError(str(parquet_dir))


def load_persrec_tc5_parquet(
    local_parquet_dir: Path,
    *,
    use_sanity_subset: bool,
    max_parts: int | None = None,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    local_parquet_dir = Path(local_parquet_dir)
    if not local_parquet_dir.exists():
        raise FileNotFoundError(f"Missing parquet directory: {str(local_parquet_dir)}")
    files = _list_parquet_part_files(local_parquet_dir)
    if not files:
        raise FileNotFoundError(f"No parquet part files found in: {str(local_parquet_dir)}")
    if bool(use_sanity_subset):
        files = [files[0]]
    elif max_parts is not None:
        m = int(max_parts)
        if m <= 0:
            raise ValueError("max_parts must be >= 1")
        files = files[:m]
    t0 = time.perf_counter()
    dfs = [pd.read_parquet(str(p)) for p in files]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info("persrec_tc5: parquet loaded rows=%d in %.3fs", int(len(df)), float(time.perf_counter() - t0))
    return df


def ensure_mapped_parquet_cache(
    *,
    source_parquet_dir: Path,
    mapped_parquet_dir: Path,
    mapped_meta_path: Path,
    product_column: str,
    max_parts: int | None = None,
) -> None:
    logger = logging.getLogger(__name__)
    source_parquet_dir = Path(source_parquet_dir)
    mapped_parquet_dir = Path(mapped_parquet_dir)
    mapped_meta_path = Path(mapped_meta_path)
    if mapped_parquet_dir.exists() and any(mapped_parquet_dir.iterdir()) and mapped_meta_path.exists():
        meta_ok = False
        try:
            z = np.load(str(mapped_meta_path))
            meta_ok = "plu_idxs" in getattr(z, "files", [])
        except Exception:
            meta_ok = False
        if meta_ok:
            logger.info(
                "persrec_tc5: using mapped parquet cache at %s (meta=%s)",
                str(mapped_parquet_dir),
                str(mapped_meta_path),
            )
            return
        logger.info(
            "persrec_tc5: mapped parquet cache at %s is missing `plu_idxs` in meta=%s -> rebuilding",
            str(mapped_parquet_dir),
            str(mapped_meta_path),
        )
        try:
            shutil.rmtree(str(mapped_parquet_dir))
        except Exception:
            pass
        try:
            mapped_meta_path.unlink(missing_ok=True)
        except Exception:
            pass
    logger.info(
        "persrec_tc5: mapped parquet cache missing/incomplete at %s (meta=%s) -> building from %s (max_parts=%s)",
        str(mapped_parquet_dir),
        str(mapped_meta_path),
        str(source_parquet_dir),
        str(max_parts),
    )

    source_files = _list_parquet_part_files(source_parquet_dir)
    if not source_files:
        raise FileNotFoundError(f"No parquet part files found in: {str(source_parquet_dir)}")
    if max_parts is not None:
        m = int(max_parts)
        if m <= 0:
            raise ValueError("max_parts must be >= 1")
        source_files = source_files[:m]

    mapped_parquet_dir.mkdir(parents=True, exist_ok=True)
    mapped_meta_path.parent.mkdir(parents=True, exist_ok=True)

    item2id: dict[int, int] = {}
    counts_list: list[int] = []
    raw_by_idx: list[int] = []
    is_plu_by_idx: list[bool] = []
    t0 = time.perf_counter()
    for part_path in tqdm(source_files, desc="persrec_tc5 map parquet", unit="part", dynamic_ncols=True, leave=False):
        df_part = pd.read_parquet(str(part_path))
        if product_column not in df_part.columns:
            raise KeyError(f"Missing product column '{product_column}' in parquet part {str(part_path)}")
        mapped_seqs: list[list[int]] = []
        for seq in df_part[product_column].tolist():
            if seq is None:
                mapped_seqs.append([])
                continue
            out: list[int] = []
            for x in list(seq):
                raw = int(x)
                idx = item2id.get(raw)
                if idx is None:
                    idx = int(len(item2id))
                    item2id[raw] = idx
                    counts_list.append(0)
                    raw_by_idx.append(int(raw))
                    is_plu_by_idx.append(int(raw) >= 0)
                out.append(int(idx))
                counts_list[int(idx)] += 1
            mapped_seqs.append(out)
        df_part = df_part.copy()
        df_part[product_column] = mapped_seqs
        df_part.to_parquet(str(mapped_parquet_dir / part_path.name), index=False)

    plu_idxs = np.nonzero(np.asarray(is_plu_by_idx, dtype=np.bool_))[0].astype(np.int64)
    np.savez(str(mapped_meta_path), counts=np.asarray(counts_list, dtype=np.int64), plu_idxs=plu_idxs)
    logger.info(
        "persrec_tc5: mapped parquet built parts=%d items=%d in %.3fs (%s)",
        int(len(source_files)),
        int(len(counts_list)),
        float(time.perf_counter() - t0),
        str(mapped_parquet_dir),
    )


def prepare_persrec_tc5_from_df(
    df: pd.DataFrame,
    *,
    base_dir: Path,
    user_column: str,
    product_column: str,
    state_size: int,
    use_sanity_subset: bool,
    seed: int,
    mapped_counts: np.ndarray | None,
) -> tuple[str, Path, Path, Dataset, Dataset, Dataset]:
    logger = logging.getLogger(__name__)
    if user_column not in df.columns:
        raise KeyError(f"Missing user column '{user_column}' in parquet dataset")
    if product_column not in df.columns:
        raise KeyError(f"Missing product column '{product_column}' in parquet dataset")

    base_dir = Path(base_dir)
    splits_path = base_dir / ("data_splits_sanity.npz" if use_sanity_subset else "data_splits.npz")
    data_statis_path = base_dir / ("data_statis_sanity.df" if use_sanity_subset else "data_statis.df")
    pop_dict_path = base_dir / ("pop_dict_sanity.txt" if use_sanity_subset else "pop_dict.txt")

    seqs: list[list[int]] = []
    for seq in df[product_column].tolist():
        if seq is None:
            seqs.append([])
        else:
            seqs.append([int(x) for x in list(seq)])

    if mapped_counts is None:
        counts_list: list[int] = []
        for s in seqs:
            for idx in s:
                i = int(idx)
                if i >= int(len(counts_list)):
                    counts_list.extend([0] * (i + 1 - int(len(counts_list))))
                counts_list[i] += 1
        mapped_counts = np.asarray(counts_list, dtype=np.int64)

    item_num_local = int(np.asarray(mapped_counts).shape[0])
    logger.info("persrec_tc5: item_num=%d", int(item_num_local))
    ensure_data_statis(data_statis_path, state_size=int(state_size), item_num=int(item_num_local))
    ensure_pop_dict(pop_dict_path, counts=np.asarray(mapped_counts, dtype=np.int64))
    train_idx, val_idx, test_idx = load_or_build_row_splits(n_rows=int(len(seqs)), splits_path=splits_path, seed=int(seed))

    data_directory = str(base_dir)
    t0 = time.perf_counter()
    train_ds = PersrecTC5UserSeqDataset(seqs, train_idx)
    val_ds = PersrecTC5UserSeqDataset(seqs, val_idx)
    test_ds = PersrecTC5UserSeqDataset(seqs, test_idx)
    logger.info("persrec_tc5: dataset objects ready in %.3fs", float(time.perf_counter() - t0))
    return data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds


def load_or_build_row_splits(*, n_rows: int, splits_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logging.getLogger(__name__)
    splits_path = Path(splits_path)
    if splits_path.exists():
        logger.info("persrec_tc5: loading splits from %s", str(splits_path))
        z = np.load(str(splits_path))
        return z["train_idx"], z["val_idx"], z["test_idx"]
    logger.info("persrec_tc5: creating splits (80/10/10) at %s", str(splits_path))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(int(n_rows))
    n_train = int(0.8 * n_rows)
    n_val = int(0.1 * n_rows)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(splits_path), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    logger.info(
        "persrec_tc5: splits sizes train=%d val=%d test=%d",
        int(train_idx.shape[0]),
        int(val_idx.shape[0]),
        int(test_idx.shape[0]),
    )
    return train_idx, val_idx, test_idx


def ensure_data_statis(path: Path, *, state_size: int, item_num: int):
    logger = logging.getLogger(__name__)
    path = Path(path)
    if path.exists():
        logger.info("persrec_tc5: data_statis exists at %s", str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"state_size": [int(state_size)], "item_num": [int(item_num)]})
    df.to_pickle(str(path))
    logger.info("persrec_tc5: wrote data_statis to %s", str(path))


def ensure_pop_dict(path: Path, *, counts: np.ndarray):
    logger = logging.getLogger(__name__)
    path = Path(path)
    if path.exists():
        logger.info("persrec_tc5: pop_dict exists at %s", str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    total = float(np.asarray(counts, dtype=np.float64).sum())
    if total <= 0:
        pop = {int(i): 0.0 for i in range(int(counts.shape[0]))}
    else:
        pop = {int(i): float(c / total) for i, c in enumerate(counts.tolist())}
    with open(path, "w") as f:
        f.write(str(pop))
    logger.info("persrec_tc5: wrote pop_dict to %s", str(path))


class PersrecTC5UserSeqDataset(Dataset):
    def __init__(self, sequences: list[list[int]], indices: np.ndarray):
        super().__init__()
        self.sequences = sequences
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        seq = self.sequences[int(self.indices[int(idx)])]
        items = torch.as_tensor(seq, dtype=torch.long)
        is_buy = torch.ones((int(items.numel()),), dtype=torch.long)
        return items, is_buy


def prepare_persrec_tc5(
    *,
    dataset_root: Path,
    data_rel: str,
    dataset_name: str,
    dataset_cfg: dict,
    seed: int,
    limit_chunks_pct: float | None = None,
) -> tuple[str, Path, Path, Dataset, Dataset, Dataset]:
    logger = logging.getLogger(__name__)
    use_sanity_subset = bool(dataset_cfg.get("use_sanity_subset", False))
    user_column = str(dataset_cfg.get("user_column", "loyalty_cardholder_rk"))
    product_column = str(dataset_cfg.get("product_column", "product_id"))
    state_size_cfg = int(dataset_cfg.get("state_size", 50))
    base_dir = Path(dataset_root) / str(data_rel) / str(dataset_name)
    logger.info("persrec_tc5: data_dir=%s sanity=%s", str(base_dir), str(bool(use_sanity_subset)))

    local_parquet_dir = base_dir / "dataset_train.parquet"
    ensure_local_parquet_cache(
        hdfs_working_prefix=str(dataset_cfg.get("hdfs_working_prefix")),
        local_parquet_dir=local_parquet_dir,
    )

    n_chunks = None
    if limit_chunks_pct is not None:
        if not (0.0 < float(limit_chunks_pct) <= 1.0):
            raise ValueError("limit_chunks_pct must be in (0, 1]")
        total = int(len(_list_parquet_part_files(local_parquet_dir)))
        if total <= 0:
            raise FileNotFoundError(f"No parquet part files found in: {str(local_parquet_dir)}")
        n_chunks = max(1, min(total, int(math.ceil(float(total) * float(limit_chunks_pct)))))
        base_dir = base_dir / f"limit_chunks={int(n_chunks)}"
        logger.info("persrec_tc5: limit_chunks_pct=%s -> n_chunks=%d data_dir=%s", str(limit_chunks_pct), int(n_chunks), str(base_dir))

    mapped_parquet_dir = base_dir / "dataset_train_mapped.parquet"
    mapped_meta_path = base_dir / "dataset_train_mapped_meta.npz"
    logger.info(
        "persrec_tc5: mapped parquet dir=%s (meta=%s) source=%s",
        str(mapped_parquet_dir),
        str(mapped_meta_path),
        str(local_parquet_dir),
    )
    ensure_mapped_parquet_cache(
        source_parquet_dir=local_parquet_dir,
        mapped_parquet_dir=mapped_parquet_dir,
        mapped_meta_path=mapped_meta_path,
        product_column=product_column,
        max_parts=n_chunks,
    )

    df = load_persrec_tc5_parquet(mapped_parquet_dir, use_sanity_subset=use_sanity_subset)
    mapped_counts = None
    if mapped_meta_path.exists():
        z = np.load(str(mapped_meta_path))
        mapped_counts = np.asarray(z["counts"], dtype=np.int64)
    return prepare_persrec_tc5_from_df(
        df,
        base_dir=base_dir,
        user_column=user_column,
        product_column=product_column,
        state_size=int(state_size_cfg),
        use_sanity_subset=use_sanity_subset,
        seed=int(seed),
        mapped_counts=mapped_counts,
    )


__all__ = [
    "prepare_persrec_tc5",
    "prepare_persrec_tc5_from_df",
    "PersrecTC5UserSeqDataset",
    "ensure_local_parquet_cache",
    "ensure_mapped_parquet_cache",
    "load_persrec_tc5_parquet",
    "load_or_build_row_splits",
    "ensure_data_statis",
    "ensure_pop_dict",
]

