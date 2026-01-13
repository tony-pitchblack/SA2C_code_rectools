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


def _pretty_dataset_name(dataset_name: str) -> str:
    ds = str(dataset_name)
    return {"yoochoose": "YooChoose", "retailrocket": "RetailRocket"}.get(ds, ds)


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


def _plot_group(
    *,
    title: str,
    dataset_name: str,
    rows: list[tuple[str, float, float, str]],
    out_path: Path,
    max_metric_value: float | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib") from e

    try:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib import patheffects as pe
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib") from e

    color_map = {"torch": "0.6", "rectools": "C3"}
    rows = list(rows)

    fig_h = max(4.5, 0.35 * max(len(rows), 1) * 2.0 + 1.6)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, fig_h), height_ratios=[1.0, 1.0, 0.55])
    if " / " in title:
        ds_title, subtitle = title.split(" / ", 1)
    else:
        ds_title, subtitle = title, None
    fig.suptitle(str(ds_title), fontsize=24, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, str(subtitle), ha="center", va="top", fontsize=14)

    notes = "\n".join(
        [
            "Notes on config names:",
            'default = "default SASRecSA2C"',
            'baseline = "baseline SASRec"',
            '*auto_warmup = "phase1 (warmup phase) early stopping on val/ndcg@10"',
            'NO *auto_warmup = "hardcoded epochs for phase1 (warmup phase)"',
            'sampled_loss = "use sampled softmax for actor & sample next-state Q-values for critic"',
            "",
            "Notes on implementations:",
            'torch = "reimplementation of author\'s code w/ torch"',
            'rectools = "reimplementation of author\'s code w/ torch + use rectools SASRec model arch"',
        ]
    )

    def add_paper_lines(ax, kind: str):
        ds = _PAPER_NDCG10.get(str(dataset_name))
        if ds is None:
            return []
        vals = ds.get(kind, {})
        if not vals:
            return []
        h1 = ax.axvline(float(vals["SASRec"]), linestyle="--", linewidth=1.2, color="C0")
        h2 = ax.axvline(
            float(vals["SASRec-SA2C"]), linestyle="--", linewidth=1.2, color="C2"
        )
        return [h1, h2]

    def barh(ax, items: list[tuple[str, float, str]], *, kind: str):
        configs = sorted({str(x[0]) for x in items})
        by_cfg: dict[str, dict[str, float]] = {}
        for cfg, val, src in items:
            by_cfg.setdefault(str(cfg), {})[str(src)] = float(val)

        y = list(range(len(configs)))
        h = 0.35
        offset = 0.20
        bars = []
        for i, cfg in enumerate(configs):
            m = by_cfg.get(cfg, {})
            if "torch" in m:
                bars.extend(
                    ax.barh(i - offset, float(m["torch"]), height=h, color=color_map["torch"])
                )
            if "rectools" in m:
                bars.extend(
                    ax.barh(i + offset, float(m["rectools"]), height=h, color=color_map["rectools"])
                )

        ax.set_yticks(y, labels=configs)
        ax.invert_yaxis()
        ax.set_xlim(left=0.0)

        paper_handles = add_paper_lines(ax, kind)
        max_line = max((float(h.get_xdata()[0]) for h in paper_handles), default=0.0)
        max_val = 0.0
        for cfg in configs:
            for v in by_cfg.get(cfg, {}).values():
                max_val = max(max_val, float(v))
        xmax_auto = max(max_val, max_line) * 1.05 if max(max_val, max_line) > 0 else 1.0
        if max_metric_value is not None and float(max_metric_value) > 0:
            xmax = float(max_metric_value)
        else:
            xmax = float(xmax_auto)
        ax.set_xlim(left=0.0, right=float(xmax))

        dx = 0.01 * float(xmax)
        text_effects = [pe.withStroke(linewidth=3.0, foreground="white", alpha=0.85)]
        for b in bars:
            w = float(b.get_width())
            ymid = float(b.get_y() + b.get_height() / 2.0)
            if w >= 0.96 * float(xmax):
                ax.text(
                    w - dx,
                    ymid,
                    f"{w:.3f}",
                    va="center",
                    ha="right",
                    fontsize=8,
                    clip_on=True,
                    path_effects=text_effects,
                )
            else:
                ax.text(
                    w + dx,
                    ymid,
                    f"{w:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    clip_on=True,
                    path_effects=text_effects,
                )

    barh(axes[0], [(cfg, p, src) for cfg, _, p, src in rows], kind="purchase")
    axes[0].set_title("purchase test/ndcg@10")

    barh(axes[1], [(cfg, c, src) for cfg, c, _, src in rows], kind="clicks")
    axes[1].set_title("clicks test/ndcg@10")

    legend_handles = [
        Patch(color=color_map["torch"], label="torch"),
        Patch(color=color_map["rectools"], label="rectools"),
        Line2D([0], [0], color="C0", linestyle="--", linewidth=1.2, label="paper SASRec"),
        Line2D([0], [0], color="C2", linestyle="--", linewidth=1.2, label="paper SASRec-SA2C"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", frameon=False)

    axes[2].set_title("Notes on configs/implementations")
    axes[2].axis("off")
    axes[2].text(0.0, 1.0, notes, va="top", ha="left", transform=axes[2].transAxes)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(str(out_path.resolve()))


def _max_metric_for_dataset(dataset_name: str, values: list[float] | None) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ds = str(dataset_name)
    if ds == "retailrocket":
        return float(values[0])
    if ds == "yoochoose":
        return float(values[1]) if len(values) > 1 else float(values[0])
    if ds.startswith("persrec_tc5_"):
        return float(values[2]) if len(values) > 2 else float(values[-1])
    return float(values[-1])


def _build_plots(
    *,
    logs_root: Path,
    only_script: str | None,
    only_dataset: str | None,
    only_eval_scheme: str | None,
    max_metric_values: list[float] | None,
):
    tqdm = _tqdm()
    by_group: dict[_GroupKey, dict[str, dict[str, tuple[float, float, float]]]] = {}

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

            mtime = max(float(clicks_path.stat().st_mtime), float(purchase_path.stat().st_mtime))
            group_map = by_group.setdefault(group_key, {})
            cfg_map = group_map.setdefault(str(config_label), {})
            prev = cfg_map.get(str(source))
            if prev is None or mtime > float(prev[2]):
                cfg_map[str(source)] = (float(clicks), float(purchase), float(mtime))

    for group_key, cfg_map in tqdm(list(by_group.items()), desc="plot", unit="dataset"):
        rows: list[tuple[str, float, float, str]] = []
        for cfg in sorted(cfg_map.keys()):
            for src in ("torch", "rectools"):
                v = cfg_map.get(cfg, {}).get(src)
                if v is None:
                    continue
                rows.append((str(cfg), float(v[0]), float(v[1]), str(src)))

        results_root = logs_root.parent / "results"
        plots_root = results_root / "plots"
        pretty_ds = _pretty_dataset_name(group_key.dataset_name)
        if group_key.eval_scheme is None:
            out_dir = plots_root / group_key.dataset_name
            title = f"{pretty_ds}"
        else:
            out_dir = plots_root / group_key.dataset_name / group_key.eval_scheme
            title = f"{pretty_ds} / {group_key.eval_scheme}"

        _plot_group(
            title=title,
            dataset_name=group_key.dataset_name,
            rows=rows,
            out_path=out_dir / "test_results.png",
            max_metric_value=_max_metric_for_dataset(group_key.dataset_name, max_metric_values),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", default=None, choices=["SA2C_SASRec_torch", "SA2C_SASRec_rectools"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--eval-scheme", default=None)
    p.add_argument(
        "--max-metric-value",
        nargs="*",
        type=float,
        default=None,
        help="Either a single float (applied to all datasets) or 3 floats: retailrocket yoochoose persrec_tc5.",
    )
    args = p.parse_args()

    logs_root = _repo_root() / "logs"
    if not logs_root.exists():
        return

    _build_plots(
        logs_root=logs_root,
        only_script=args.script,
        only_dataset=args.dataset,
        only_eval_scheme=args.eval_scheme,
        max_metric_values=args.max_metric_value,
    )


if __name__ == "__main__":
    main()

