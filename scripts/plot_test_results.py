from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _GroupKey:
    script_name: str
    dataset_name: str
    eval_scheme: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_persrec_tc5_dataset(dataset_name: str) -> bool:
    return str(dataset_name).startswith("persrec_tc5_")


def _iter_result_pairs(root: Path):
    for clicks_path in root.rglob("results_clicks.csv"):
        run_dir = clicks_path.parent
        purchase_path = run_dir / "results_purchase.csv"
        if not purchase_path.exists():
            continue
        yield run_dir, clicks_path, purchase_path


def _extract_group_and_config(*, script_name: str, script_root: Path, run_dir: Path):
    rel = run_dir.relative_to(script_root)
    parts = rel.parts
    if len(parts) < 2:
        return None

    dataset = str(parts[0])
    if script_name == "SA2C_SASRec_rectools" and _is_persrec_tc5_dataset(dataset):
        if len(parts) < 3:
            return None
        eval_scheme = str(parts[1])
        config_label = "/".join(parts[2:])
        group_key = _GroupKey(script_name=script_name, dataset_name=dataset, eval_scheme=eval_scheme)
        out_dir = script_root / dataset / eval_scheme
        return group_key, config_label, out_dir

    config_label = "/".join(parts[1:])
    group_key = _GroupKey(script_name=script_name, dataset_name=dataset, eval_scheme=None)
    out_dir = script_root / dataset
    return group_key, config_label, out_dir


def _read_test_ndcg_at_10(csv_path: Path) -> float | None:
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pandas") from e

    df = pd.read_csv(csv_path)
    if "test/ndcg@10" not in df.columns or df.shape[0] < 1:
        return None
    v = df.loc[df.index[0], "test/ndcg@10"]
    try:
        return float(v)
    except Exception:
        return None


def _plot_group(*, title: str, rows: list[tuple[str, float, float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib") from e

    rows_p = sorted(rows, key=lambda x: float(x[2]), reverse=True)
    rows_c = sorted(rows, key=lambda x: float(x[1]), reverse=True)

    fig_h = max(3.0, 0.35 * max(len(rows), 1) * 2.0)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, fig_h))
    fig.suptitle(title)

    def barh(ax, items: list[tuple[str, float]]):
        labels = [x[0] for x in items]
        vals = [float(x[1]) for x in items]
        y = list(range(len(labels)))
        ax.barh(y, vals)
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0.0)

    barh(axes[0], [(cfg, p) for cfg, _, p in rows_p])
    axes[0].set_title("purchase test/ndcg@10")

    barh(axes[1], [(cfg, c) for cfg, c, _ in rows_c])
    axes[1].set_title("clicks test/ndcg@10")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_plots(*, logs_root: Path, only_script: str | None, only_dataset: str | None, only_eval_scheme: str | None):
    by_group: dict[_GroupKey, dict[str, tuple[float, float, float]]] = {}

    for script_name in ("SA2C_SASRec_torch", "SA2C_SASRec_rectools"):
        if only_script is not None and script_name != only_script:
            continue
        script_root = logs_root / script_name
        if not script_root.exists():
            continue

        for run_dir, clicks_path, purchase_path in _iter_result_pairs(script_root):
            parsed = _extract_group_and_config(script_name=script_name, script_root=script_root, run_dir=run_dir)
            if parsed is None:
                continue
            group_key, config_label, _ = parsed
            if only_dataset is not None and group_key.dataset_name != only_dataset:
                continue
            if only_eval_scheme is not None and group_key.eval_scheme != only_eval_scheme:
                continue

            clicks = _read_test_ndcg_at_10(clicks_path)
            purchase = _read_test_ndcg_at_10(purchase_path)
            if clicks is None or purchase is None:
                continue

            mtime = max(float(clicks_path.stat().st_mtime), float(purchase_path.stat().st_mtime))
            group_map = by_group.setdefault(group_key, {})
            prev = group_map.get(config_label)
            if prev is None or mtime > float(prev[2]):
                group_map[config_label] = (float(clicks), float(purchase), float(mtime))

    for group_key, cfg_map in by_group.items():
        rows = [(cfg, v[0], v[1]) for cfg, v in cfg_map.items()]
        rows.sort(key=lambda x: x[0])

        group_script_root = logs_root / group_key.script_name
        if group_key.eval_scheme is None:
            out_dir = group_script_root / group_key.dataset_name
            title = f"{group_key.script_name} / {group_key.dataset_name}"
        else:
            out_dir = group_script_root / group_key.dataset_name / group_key.eval_scheme
            title = f"{group_key.script_name} / {group_key.dataset_name} / {group_key.eval_scheme}"

        _plot_group(title=title, rows=rows, out_path=out_dir / "test_results.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", default=None, choices=["SA2C_SASRec_torch", "SA2C_SASRec_rectools"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--eval-scheme", default=None)
    args = p.parse_args()

    logs_root = _repo_root() / "logs"
    if not logs_root.exists():
        return

    _build_plots(
        logs_root=logs_root,
        only_script=args.script,
        only_dataset=args.dataset,
        only_eval_scheme=args.eval_scheme,
    )


if __name__ == "__main__":
    main()

