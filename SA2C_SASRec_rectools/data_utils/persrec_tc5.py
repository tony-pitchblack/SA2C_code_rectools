from __future__ import annotations

import logging
import pickle
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
        logger.info("persrec_tc5: parquet cache found at %s", str(local_parquet_dir))
        return
    local_parquet_dir.parent.mkdir(parents=True, exist_ok=True)
    src = str(Path(hdfs_working_prefix) / "training" / "dataset_train.parquet")
    dst = str(local_parquet_dir)
    t0 = time.perf_counter()
    logger.info("persrec_tc5: parquet cache missing -> downloading to %s", str(local_parquet_dir))
    hdfs_get(src, dst)
    logger.info("persrec_tc5: hdfs download done in %.3fs", float(time.perf_counter() - t0))
    if not local_parquet_dir.exists():
        raise RuntimeError(f"HDFS download completed but local path does not exist: {str(local_parquet_dir)}")


def load_persrec_tc5_parquet(local_parquet_dir: Path, *, use_sanity_subset: bool) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    local_parquet_dir = Path(local_parquet_dir)
    if not local_parquet_dir.exists():
        raise FileNotFoundError(f"Missing parquet directory: {str(local_parquet_dir)}")
    if not bool(use_sanity_subset):
        t0 = time.perf_counter()
        logger.info("persrec_tc5: loading parquet dataset (all parts) from %s", str(local_parquet_dir))
        df = pd.read_parquet(str(local_parquet_dir))
        logger.info("persrec_tc5: parquet loaded rows=%d in %.3fs", int(len(df)), float(time.perf_counter() - t0))
        return df
    files = sorted(
        [p for p in local_parquet_dir.iterdir() if p.is_file() and p.suffix == ".parquet"], key=lambda p: p.name
    )
    if not files:
        raise FileNotFoundError(f"Sanity subset requested, but no parquet part files found in: {str(local_parquet_dir)}")
    t0 = time.perf_counter()
    logger.info("persrec_tc5: sanity subset -> loading parquet part %s", str(files[0]))
    df = pd.read_parquet(str(files[0]))
    logger.info("persrec_tc5: parquet loaded rows=%d in %.3fs", int(len(df)), float(time.perf_counter() - t0))
    return df


def prepare_persrec_tc5_from_df(
    df: pd.DataFrame,
    *,
    base_dir: Path,
    user_column: str,
    product_column: str,
    state_size: int,
    use_sanity_subset: bool,
    seed: int,
) -> tuple[str, Path, Path, Dataset, Dataset, Dataset]:
    logger = logging.getLogger(__name__)
    if user_column not in df.columns:
        raise KeyError(f"Missing user column '{user_column}' in parquet dataset")
    if product_column not in df.columns:
        raise KeyError(f"Missing product column '{product_column}' in parquet dataset")

    base_dir = Path(base_dir)
    vocab_path = base_dir / ("built_vocabulary_sanity.npz" if use_sanity_subset else "built_vocabulary.pkl")
    splits_path = base_dir / ("data_splits_sanity.npz" if use_sanity_subset else "data_splits.npz")
    data_statis_path = base_dir / ("data_statis_sanity.df" if use_sanity_subset else "data_statis.df")
    pop_dict_path = base_dir / ("pop_dict_sanity.txt" if use_sanity_subset else "pop_dict.txt")

    seqs, item2id, counts = load_or_build_item_vocab_and_sequences(
        df, product_column=product_column, vocab_path=vocab_path, use_sanity_subset=use_sanity_subset
    )
    item_num_local = int(len(item2id))
    logger.info("persrec_tc5: item_num=%d", int(item_num_local))
    ensure_data_statis(data_statis_path, state_size=int(state_size), item_num=int(item_num_local))
    ensure_pop_dict(pop_dict_path, counts=counts)
    train_idx, val_idx, test_idx = load_or_build_row_splits(n_rows=int(len(seqs)), splits_path=splits_path, seed=int(seed))

    data_directory = str(base_dir)
    t0 = time.perf_counter()
    train_ds = PersrecTC5UserSeqDataset(seqs, train_idx)
    val_ds = PersrecTC5UserSeqDataset(seqs, val_idx)
    test_ds = PersrecTC5UserSeqDataset(seqs, test_idx)
    logger.info("persrec_tc5: dataset objects ready in %.3fs", float(time.perf_counter() - t0))
    return data_directory, data_statis_path, pop_dict_path, train_ds, val_ds, test_ds


def load_vocab_any(path: Path, *, is_sanity: bool) -> dict[int, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if bool(is_sanity):
        z = np.load(str(path))
        raw_ids = z["raw_ids"]
        return {int(raw): int(i) for i, raw in enumerate(raw_ids.tolist())}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in vocab file, got {type(obj)}")
    return {int(k): int(v) for k, v in obj.items()}


def save_vocab_any(path: Path, item2id: dict[int, int], *, is_sanity: bool):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if bool(is_sanity):
        raw_ids = np.empty((len(item2id),), dtype=np.int64)
        for raw, idx in item2id.items():
            raw_ids[int(idx)] = int(raw)
        np.savez(str(path), raw_ids=raw_ids)
        return
    with open(path, "wb") as f:
        pickle.dump(item2id, f)


def load_or_build_item_vocab_and_sequences(
    df: pd.DataFrame,
    *,
    product_column: str,
    vocab_path: Path,
    use_sanity_subset: bool,
) -> tuple[list[list[int]], dict[int, int], np.ndarray]:
    logger = logging.getLogger(__name__)
    is_sanity = bool(use_sanity_subset)
    t0 = time.perf_counter()
    if Path(vocab_path).exists():
        logger.info("persrec_tc5: loading vocab from %s", str(vocab_path))
        item2id = load_vocab_any(vocab_path, is_sanity=is_sanity)
        counts = np.zeros((len(item2id),), dtype=np.int64)
        seqs = []
        for seq in tqdm(
            df[product_column].tolist(), desc="persrec_tc5 remap", unit="row", dynamic_ncols=True, leave=False
        ):
            if seq is None:
                seqs.append([])
                continue
            remapped = []
            for x in list(seq):
                raw = int(x)
                if raw not in item2id:
                    raise ValueError(f"Unknown product_id={raw} not in existing vocabulary {str(vocab_path)}")
                idx = int(item2id[raw])
                remapped.append(idx)
                counts[idx] += 1
            seqs.append(remapped)
        logger.info(
            "persrec_tc5: remapped sequences rows=%d vocab=%d in %.3fs",
            int(len(seqs)),
            int(len(item2id)),
            float(time.perf_counter() - t0),
        )
        return seqs, item2id, counts

    logger.info("persrec_tc5: building vocab at %s", str(vocab_path))
    item2id: dict[int, int] = {}
    counts_list: list[int] = []
    seqs = []
    for seq in tqdm(df[product_column].tolist(), desc="persrec_tc5 vocab", unit="row", dynamic_ncols=True, leave=False):
        if seq is None:
            seqs.append([])
            continue
        remapped = []
        for x in list(seq):
            raw = int(x)
            idx = item2id.get(raw)
            if idx is None:
                idx = int(len(item2id))
                item2id[raw] = idx
                counts_list.append(0)
            remapped.append(idx)
            counts_list[idx] += 1
        seqs.append(remapped)
    counts = np.asarray(counts_list, dtype=np.int64)
    save_vocab_any(vocab_path, item2id, is_sanity=is_sanity)
    logger.info(
        "persrec_tc5: built vocab rows=%d vocab=%d in %.3fs",
        int(len(seqs)),
        int(len(item2id)),
        float(time.perf_counter() - t0),
    )
    return seqs, item2id, counts


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
    df = load_persrec_tc5_parquet(local_parquet_dir, use_sanity_subset=use_sanity_subset)
    logger.info("persrec_tc5: parquet columns=%s", ",".join([str(c) for c in df.columns.tolist()]))
    return prepare_persrec_tc5_from_df(
        df,
        base_dir=base_dir,
        user_column=user_column,
        product_column=product_column,
        state_size=int(state_size_cfg),
        use_sanity_subset=use_sanity_subset,
        seed=int(seed),
    )


__all__ = [
    "prepare_persrec_tc5",
    "prepare_persrec_tc5_from_df",
    "PersrecTC5UserSeqDataset",
    "ensure_local_parquet_cache",
    "load_persrec_tc5_parquet",
    "load_or_build_item_vocab_and_sequences",
    "load_or_build_row_splits",
    "ensure_data_statis",
    "ensure_pop_dict",
]

