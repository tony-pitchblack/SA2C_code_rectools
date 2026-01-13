from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


_PAPER_NDCG10 = {
    "retailrocket": {
        "purchase": {"SASRec": 0.4510, "SASRec-SA2C": 0.5246},
        "clicks": {"SASRec": 0.2107, "SASRec-SA2C": 0.2416},
    },
    "yoochoose": {
        "purchase": {"SASRec": 0.3326, "SASRec-SA2C": 0.3728},
        "clicks": {"SASRec": 0.2515, "SASRec-SA2C": 0.2719},
    },
}


@dataclass(frozen=True)
class _GroupKey:
    dataset_name: str
    eval_scheme: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: tqdm") from e
    return tqdm


def _is_persrec_tc5_dataset(dataset_name: str) -> bool:
    return str(dataset_name).startswith("persrec_tc5_")


def _iter_result_pairs(root: Path):
    tqdm = _tqdm()
    clicks_paths = list(root.rglob("results_clicks.csv"))
    for clicks_path in tqdm(clicks_paths, desc=f"scan {root.name}", unit="run"):
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
        group_key = _GroupKey(dataset_name=dataset, eval_scheme=eval_scheme)
        return group_key, config_label

    config_label = "/".join(parts[1:])
    group_key = _GroupKey(dataset_name=dataset, eval_scheme=None)
    return group_key, config_label


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


def _plot_group(*, title: str, dataset_name: str, rows: list[tuple[str, float, float, str]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib") from e

    try:
        from matplotlib.patches import Patch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib") from e

    color_map = {"torch": "0.6", "rectools": "C3"}
    rows_p = sorted(rows, key=lambda x: float(x[2]), reverse=True)
    rows_c = sorted(rows, key=lambda x: float(x[1]), reverse=True)

    fig_h = max(3.0, 0.35 * max(len(rows), 1) * 2.0)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, fig_h))
    fig.suptitle(title)

    def add_paper_lines(ax, kind: str):
        ds = _PAPER_NDCG10.get(str(dataset_name))
        if ds is None:
            return []
        vals = ds.get(kind, {})
        if not vals:
            return []
        h1 = ax.axvline(float(vals["SASRec"]), linestyle="--", linewidth=1.2, color="C0", label="paper SASRec")
        h2 = ax.axvline(
            float(vals["SASRec-SA2C"]), linestyle="--", linewidth=1.2, color="C2", label="paper SASRec-SA2C"
        )
        return [h1, h2]

    def barh(ax, items: list[tuple[str, float, str]], *, kind: str):
        labels = [x[0] for x in items]
        vals = [float(x[1]) for x in items]
        colors = [color_map.get(str(x[2]), "C7") for x in items]
        y = list(range(len(labels)))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0.0)

        legend_handles = []
        present_sources = {str(x[2]) for x in items}
        if "torch" in present_sources:
            legend_handles.append(Patch(color=color_map["torch"], label="torch"))
        if "rectools" in present_sources:
            legend_handles.append(Patch(color=color_map["rectools"], label="rectools"))

        paper_handles = add_paper_lines(ax, kind)
        legend_handles.extend(paper_handles)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower right", frameon=False)

        max_val = max(vals) if vals else 0.0
        max_line = max((float(h.get_xdata()[0]) for h in paper_handles), default=0.0)
        xmax = max(max_val, max_line) * 1.05 if max(max_val, max_line) > 0 else 1.0
        ax.set_xlim(left=0.0, right=float(xmax))

    barh(axes[0], [(cfg, p, src) for cfg, _, p, src in rows_p], kind="purchase")
    axes[0].set_title("purchase test/ndcg@10")

    barh(axes[1], [(cfg, c, src) for cfg, c, _, src in rows_c], kind="clicks")
    axes[1].set_title("clicks test/ndcg@10")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(str(out_path.resolve()))


def _build_plots(*, logs_root: Path, only_script: str | None, only_dataset: str | None, only_eval_scheme: str | None):
    tqdm = _tqdm()
    by_group: dict[_GroupKey, dict[str, tuple[float, float, float, str]]] = {}

    for script_name in ("SA2C_SASRec_torch", "SA2C_SASRec_rectools"):
        if only_script is not None and script_name != only_script:
            continue
        script_root = logs_root / script_name
        if not script_root.exists():
            continue

        source = "torch" if script_name == "SA2C_SASRec_torch" else "rectools"
        for run_dir, clicks_path, purchase_path in _iter_result_pairs(script_root):
            parsed = _extract_group_and_config(script_name=script_name, script_root=script_root, run_dir=run_dir)
            if parsed is None:
                continue
            group_key, config_label = parsed
            if only_dataset is not None and group_key.dataset_name != only_dataset:
                continue
            if only_eval_scheme is not None and group_key.eval_scheme != only_eval_scheme:
                continue

            clicks = _read_test_ndcg_at_10(clicks_path)
            purchase = _read_test_ndcg_at_10(purchase_path)
            if clicks is None or purchase is None:
                continue

            label = f"{source}:{config_label}"
            mtime = max(float(clicks_path.stat().st_mtime), float(purchase_path.stat().st_mtime))
            group_map = by_group.setdefault(group_key, {})
            prev = group_map.get(label)
            if prev is None or mtime > float(prev[2]):
                group_map[label] = (float(clicks), float(purchase), float(mtime), source)

    for group_key, cfg_map in tqdm(list(by_group.items()), desc="plot", unit="dataset"):
        rows = [(cfg, v[0], v[1], v[3]) for cfg, v in cfg_map.items()]
        rows.sort(key=lambda x: x[0])

        plots_root = logs_root / "plots"
        if group_key.eval_scheme is None:
            out_dir = plots_root / group_key.dataset_name
            title = f"{group_key.dataset_name}"
        else:
            out_dir = plots_root / group_key.dataset_name / group_key.eval_scheme
            title = f"{group_key.dataset_name} / {group_key.eval_scheme}"

        _plot_group(title=title, dataset_name=group_key.dataset_name, rows=rows, out_path=out_dir / "test_results.png")


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

