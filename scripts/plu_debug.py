from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(str(path))
    return {k: np.asarray(z[k]) for k in getattr(z, "files", [])}


def _describe_plu_meta(root: Path) -> None:
    meta_path = root / "dataset_train_mapped_meta.npz"
    if not meta_path.exists():
        print(f"missing: {meta_path}")
        return
    z = _load_npz(meta_path)
    counts = z.get("counts")
    plu_idxs = z.get("plu_idxs")
    if counts is None:
        print(f"invalid meta (no counts): {meta_path}")
        return
    item_num = int(np.asarray(counts).shape[0])
    if plu_idxs is None:
        print(f"meta has no plu_idxs: {meta_path}")
        return
    plu_idxs = np.asarray(plu_idxs, dtype=np.int64)
    plu_count = int(plu_idxs.size)
    pct = 0.0 if item_num <= 0 else 100.0 * float(plu_count) / float(item_num)
    print(f"item_num={item_num}")
    print(f"plu_idxs={plu_count} ({pct:.6f}%)")
    print(f"plu_filter_noop={'yes' if plu_count == item_num else 'no'}")


def _compare_splits(root: Path) -> None:
    a = root / "bert4rec_eval" / "dataset_splits.npz"
    b = root / "bert4rec_eval_plu" / "dataset_splits.npz"
    if (not a.exists()) or (not b.exists()):
        print(f"skip compare: missing {a if not a.exists() else b}")
        return
    za = _load_npz(a)
    zb = _load_npz(b)
    keys = ["train_idx", "val_idx", "test_idx"]
    if not all(k in za for k in keys) or not all(k in zb for k in keys):
        print("skip compare: missing expected keys in one of the split files")
        return
    for k in keys:
        xa = np.asarray(za[k], dtype=np.int64)
        xb = np.asarray(zb[k], dtype=np.int64)
        eq = bool(np.array_equal(xa, xb))
        print(f"{k}: equal={eq} size_a={int(xa.size)} size_b={int(xb.size)}")
        if not eq:
            sa = set(int(x) for x in xa.tolist())
            sb = set(int(x) for x in xb.tolist())
            inter = int(len(sa & sb))
            only_a = int(len(sa - sb))
            only_b = int(len(sb - sa))
            print(f"{k}: intersection={inter} only_a={only_a} only_b={only_b}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path like ~/data/persrec_tc5_2025-08-21[/limit_chunks=<n>]")
    args = p.parse_args()
    root = Path(str(args.root)).expanduser().resolve()
    print(f"root={root}")
    _describe_plu_meta(root)
    _compare_splits(root)


if __name__ == "__main__":
    main()

