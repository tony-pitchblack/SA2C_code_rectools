from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .persrec_tc5 import (
    ensure_data_statis,
    ensure_local_parquet_cache,
    ensure_mapped_parquet_cache,
    ensure_pop_dict,
    load_persrec_tc5_parquet,
)
from .sessions import SessionDatasetFromDF


def load_or_build_bert4rec_splits(
    *,
    n_rows: int,
    eligible_val_idx: np.ndarray,
    eligible_test_idx: np.ndarray | None = None,
    val_samples_num: int,
    test_samples_num: int,
    seed: int,
    splits_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits_path = Path(splits_path)
    if splits_path.exists():
        z = np.load(str(splits_path))
        return z["train_idx"], z["val_idx"], z["test_idx"]

    n_rows = int(n_rows)
    eligible_val_idx = np.asarray(eligible_val_idx, dtype=np.int64)
    if eligible_val_idx.ndim != 1:
        raise ValueError("eligible_val_idx must be 1D")
    if eligible_val_idx.size == 0:
        raise ValueError("No eligible validation sequences for bert4rec_loo splits")

    eligible_test_idx = eligible_val_idx if eligible_test_idx is None else np.asarray(eligible_test_idx, dtype=np.int64)
    if eligible_test_idx.ndim != 1:
        raise ValueError("eligible_test_idx must be 1D")
    if eligible_test_idx.size == 0:
        raise ValueError("No eligible test sequences for bert4rec_loo splits")

    eligible_val_idx = np.unique(eligible_val_idx)
    eligible_test_idx = np.unique(eligible_test_idx)

    val_samples_num = int(val_samples_num)
    test_samples_num = int(test_samples_num)
    if val_samples_num <= 0 or test_samples_num <= 0:
        raise ValueError("val_samples_num and test_samples_num must be > 0 for bert4rec_loo")

    rng = np.random.default_rng(int(seed))
    if int(test_samples_num) > int(eligible_test_idx.size):
        raise ValueError(f"test_samples_num={test_samples_num} exceeds eligible_test={int(eligible_test_idx.size)}")
    test_idx = rng.choice(eligible_test_idx, size=int(test_samples_num), replace=False)

    remaining_val = np.setdiff1d(eligible_val_idx, test_idx, assume_unique=False)
    if int(val_samples_num) > int(remaining_val.size):
        raise ValueError(f"val_samples_num={val_samples_num} exceeds eligible_val_minus_test={int(remaining_val.size)}")
    val_idx = rng.choice(remaining_val, size=int(val_samples_num), replace=False)

    all_idx = np.arange(n_rows, dtype=np.int64)
    used = np.union1d(val_idx, test_idx)
    train_idx = np.setdiff1d(all_idx, used, assume_unique=False)

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(splits_path), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return train_idx, val_idx, test_idx


class _PersrecBert4RecTrainDataset(Dataset):
    def __init__(self, sequences: list[list[int]], *, val_idx: np.ndarray):
        super().__init__()
        self.sequences = sequences
        val_mask = np.zeros((int(len(sequences)),), dtype=np.bool_)
        val_mask[np.asarray(val_idx, dtype=np.int64)] = True
        self.val_mask = val_mask

    def __len__(self):
        return int(len(self.sequences))

    def __getitem__(self, idx: int):
        seq = self.sequences[int(idx)]
        n_drop = 2 if bool(self.val_mask[int(idx)]) else 1
        if len(seq) <= n_drop:
            items = torch.empty((0,), dtype=torch.long)
        else:
            items = torch.as_tensor(seq[:-n_drop], dtype=torch.long)
        is_buy = torch.ones((int(items.numel()),), dtype=torch.long)
        return items, is_buy


class _PersrecBert4RecEvalDataset(Dataset):
    def __init__(self, sequences: list[list[int]], *, indices: np.ndarray, drop_last: int):
        super().__init__()
        self.sequences = sequences
        self.indices = np.asarray(indices, dtype=np.int64)
        self.drop_last = int(drop_last)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        seq = self.sequences[int(self.indices[int(idx)])]
        if int(self.drop_last) > 0:
            seq = seq[: -int(self.drop_last)]
        items = torch.as_tensor(seq, dtype=torch.long)
        is_buy = torch.ones((int(items.numel()),), dtype=torch.long)
        return items, is_buy


def prepare_persrec_tc5_bert4rec_loo(
    *,
    dataset_root: Path,
    data_rel: str,
    dataset_name: str,
    dataset_cfg: dict,
    seed: int,
    val_samples_num: int,
    test_samples_num: int,
    plu_filter: str,
    limit_chunks_pct: float | None = None,
) -> tuple[str, Path, Path, Dataset, Dataset, Dataset]:
    use_sanity_subset = bool(dataset_cfg.get("use_sanity_subset", False))
    product_column = str(dataset_cfg.get("product_column", "product_id"))
    state_size_cfg = int(dataset_cfg.get("state_size", 50))
    base_dir = Path(dataset_root) / str(data_rel) / str(dataset_name)

    local_parquet_dir = base_dir / "dataset_train.parquet"
    ensure_local_parquet_cache(
        hdfs_working_prefix=str(dataset_cfg.get("hdfs_working_prefix")),
        local_parquet_dir=local_parquet_dir,
    )

    max_parts = None
    if limit_chunks_pct is not None:
        if not (0.0 < float(limit_chunks_pct) <= 1.0):
            raise ValueError("limit_chunks_pct must be in (0, 1]")
        source_files = [p for p in local_parquet_dir.iterdir() if p.is_file() and p.suffix == ".parquet"]
        total = int(len(source_files))
        if total <= 0:
            raise FileNotFoundError(f"No parquet part files found in: {str(local_parquet_dir)}")
        n_chunks = max(1, min(total, int(math.ceil(float(total) * float(limit_chunks_pct)))))
        base_dir = base_dir / f"limit_chunks={int(n_chunks)}"
        max_parts = int(n_chunks)

    mapped_parquet_dir = base_dir / "dataset_train_mapped.parquet"
    mapped_meta_path = base_dir / "dataset_train_mapped_meta.npz"
    ensure_mapped_parquet_cache(
        source_parquet_dir=local_parquet_dir,
        mapped_parquet_dir=mapped_parquet_dir,
        mapped_meta_path=mapped_meta_path,
        product_column=product_column,
        max_parts=max_parts,
    )
    df = load_persrec_tc5_parquet(mapped_parquet_dir, use_sanity_subset=use_sanity_subset)

    data_statis_path = base_dir / ("data_statis_sanity.df" if use_sanity_subset else "data_statis.df")
    pop_dict_path = base_dir / ("pop_dict_sanity.txt" if use_sanity_subset else "pop_dict.txt")

    seqs: list[list[int]] = []
    for seq in df[product_column].tolist():
        if seq is None:
            seqs.append([])
        else:
            seqs.append([int(x) for x in list(seq)])

    counts = None
    plu_idxs = None
    if mapped_meta_path.exists():
        z = np.load(str(mapped_meta_path))
        counts = np.asarray(z["counts"], dtype=np.int64)
        if "plu_idxs" in getattr(z, "files", []):
            plu_idxs = np.asarray(z["plu_idxs"], dtype=np.int64)
    if counts is None:
        counts_list: list[int] = []
        for s in seqs:
            for idx in s:
                i = int(idx)
                if i >= int(len(counts_list)):
                    counts_list.extend([0] * (i + 1 - int(len(counts_list))))
                counts_list[i] += 1
        counts = np.asarray(counts_list, dtype=np.int64)

    ensure_data_statis(data_statis_path, state_size=int(state_size_cfg), item_num=int(counts.shape[0]))
    ensure_pop_dict(pop_dict_path, counts=np.asarray(counts, dtype=np.int64))

    mode = str(plu_filter).strip().lower()
    if mode not in {"enable", "disable", "inverse"}:
        raise ValueError("plu_filter must be one of: enable | disable | inverse")

    if mode == "disable":
        eligible = np.asarray([i for i, s in enumerate(seqs) if (len(s) >= 3)], dtype=np.int64)
        eligible_test = eligible
        eligible_val = eligible
        splits_dir = "bert4rec_eval"
    else:
        if plu_idxs is None:
            raise RuntimeError(f"Missing `plu_idxs` in mapped meta: {str(mapped_meta_path)}")
        plu_set = set(int(x) for x in np.asarray(plu_idxs, dtype=np.int64).tolist())
        if mode == "enable":
            eligible_test = np.asarray(
                [i for i, s in enumerate(seqs) if (len(s) >= 3 and int(s[-1]) in plu_set)],
                dtype=np.int64,
            )
            eligible_val = np.asarray(
                [i for i, s in enumerate(seqs) if (len(s) >= 3 and int(s[-2]) in plu_set)],
                dtype=np.int64,
            )
            splits_dir = "bert4rec_eval_plu"
        else:
            eligible_test = np.asarray(
                [i for i, s in enumerate(seqs) if (len(s) >= 3 and int(s[-1]) not in plu_set)],
                dtype=np.int64,
            )
            eligible_val = np.asarray(
                [i for i, s in enumerate(seqs) if (len(s) >= 3 and int(s[-2]) not in plu_set)],
                dtype=np.int64,
            )
            splits_dir = "bert4rec_eval_nonplu"

    splits_path = base_dir / str(splits_dir) / ("dataset_splits_sanity.npz" if use_sanity_subset else "dataset_splits.npz")
    train_idx, val_idx, test_idx = load_or_build_bert4rec_splits(
        n_rows=int(len(seqs)),
        eligible_val_idx=eligible_val,
        eligible_test_idx=eligible_test,
        val_samples_num=int(val_samples_num),
        test_samples_num=int(test_samples_num),
        seed=int(seed),
        splits_path=splits_path,
    )
    _ = train_idx

    train_ds = _PersrecBert4RecTrainDataset(seqs, val_idx=val_idx)
    val_ds = _PersrecBert4RecEvalDataset(seqs, indices=val_idx, drop_last=1)
    test_ds = _PersrecBert4RecEvalDataset(seqs, indices=test_idx, drop_last=0)
    return str(base_dir), data_statis_path, pop_dict_path, train_ds, val_ds, test_ds


def prepare_sessions_bert4rec_loo(
    *,
    data_directory: str,
    split_df_names: list[str],
    seed: int,
    val_samples_num: int,
    test_samples_num: int,
    limit_chunks_pct: float | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    dfs = [pd.read_pickle(str(Path(data_directory) / n)) for n in list(split_df_names)]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    base_ds = SessionDatasetFromDF(df)

    items_list = base_ds.items_list
    is_buy_list = base_ds.is_buy_list
    splits_root = Path(data_directory)
    if limit_chunks_pct is not None:
        if not (0.0 < float(limit_chunks_pct) <= 1.0):
            raise ValueError("limit_chunks_pct must be in (0, 1]")
        total = int(len(items_list))
        if total <= 0:
            raise ValueError("No sessions found")
        n_keep = max(1, min(total, int(math.ceil(float(total) * float(limit_chunks_pct)))))
        splits_root = splits_root / f"limit_chunks={int(n_keep)}"
        items_list = list(items_list[: int(n_keep)])
        is_buy_list = list(is_buy_list[: int(n_keep)])

    eligible = np.asarray([i for i, x in enumerate(items_list) if int(x.numel()) >= 3], dtype=np.int64)
    splits_path = splits_root / "bert4rec_eval" / "dataset_splits.npz"
    train_idx, val_idx, test_idx = load_or_build_bert4rec_splits(
        n_rows=int(len(items_list)),
        eligible_val_idx=eligible,
        eligible_test_idx=eligible,
        val_samples_num=int(val_samples_num),
        test_samples_num=int(test_samples_num),
        seed=int(seed),
        splits_path=splits_path,
    )
    _ = train_idx

    val_mask = np.zeros((int(len(items_list)),), dtype=np.bool_)
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True

    class _Train(Dataset):
        def __len__(self):
            return int(len(items_list))

        def __getitem__(self, idx: int):
            items = items_list[int(idx)]
            is_buy = is_buy_list[int(idx)]
            n_drop = 2 if bool(val_mask[int(idx)]) else 1
            if int(items.numel()) <= int(n_drop):
                return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
            return items[: -int(n_drop)], is_buy[: -int(n_drop)]

    class _Eval(Dataset):
        def __init__(self, indices: np.ndarray, drop_last: int):
            self.indices = np.asarray(indices, dtype=np.int64)
            self.drop_last = int(drop_last)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, i: int):
            idx = int(self.indices[int(i)])
            items = items_list[idx]
            is_buy = is_buy_list[idx]
            if int(self.drop_last) > 0:
                items = items[: -int(self.drop_last)]
                is_buy = is_buy[: -int(self.drop_last)]
            return items, is_buy

    train_ds = _Train()
    val_ds = _Eval(val_idx, drop_last=1)
    test_ds = _Eval(test_idx, drop_last=0)
    return train_ds, val_ds, test_ds


__all__ = [
    "load_or_build_bert4rec_splits",
    "prepare_persrec_tc5_bert4rec_loo",
    "prepare_sessions_bert4rec_loo",
]

