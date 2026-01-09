from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import metrics_row, overall_row, summary_at_k_text


def write_results(
    *,
    run_dir: Path,
    val_best: dict,
    test_best: dict,
    smoke_cpu: bool,
    val_warmup: dict | None = None,
    test_warmup: dict | None = None,
):
    val_click = metrics_row(val_best, "click")
    test_click = metrics_row(test_best, "click")
    val_purchase = metrics_row(val_best, "purchase")
    test_purchase = metrics_row(test_best, "purchase")
    val_overall = overall_row(val_best)
    test_overall = overall_row(test_best)

    col_order = []
    for k in val_best["topk"]:
        col_order.extend([f"val/hr@{k}", f"test/hr@{k}", f"val/ndcg@{k}", f"test/ndcg@{k}"])

    click_row = {}
    purchase_row = {}
    for k in val_best["topk"]:
        click_row[f"val/hr@{k}"] = float(val_click.get(f"hr@{k}", 0.0))
        click_row[f"test/hr@{k}"] = float(test_click.get(f"hr@{k}", 0.0))
        click_row[f"val/ndcg@{k}"] = float(val_click.get(f"ndcg@{k}", 0.0))
        click_row[f"test/ndcg@{k}"] = float(test_click.get(f"ndcg@{k}", 0.0))

        purchase_row[f"val/hr@{k}"] = float(val_purchase.get(f"hr@{k}", 0.0))
        purchase_row[f"test/hr@{k}"] = float(test_purchase.get(f"hr@{k}", 0.0))
        purchase_row[f"val/ndcg@{k}"] = float(val_purchase.get(f"ndcg@{k}", 0.0))
        purchase_row[f"test/ndcg@{k}"] = float(test_purchase.get(f"ndcg@{k}", 0.0))

    if not smoke_cpu:
        df_clicks = pd.DataFrame([click_row], index=["metrics"]).loc[:, col_order]
        df_purchase = pd.DataFrame([purchase_row], index=["metrics"]).loc[:, col_order]
        df_clicks.to_csv(run_dir / "results_clicks.csv", index=False)
        df_purchase.to_csv(run_dir / "results_purchase.csv", index=False)

    overall_col_order = []
    for k in val_best["topk"]:
        overall_col_order.extend([f"val/ndcg@{k}", f"test/ndcg@{k}"])
    overall_row_out = {}
    for k in val_best["topk"]:
        overall_row_out[f"val/ndcg@{k}"] = float(val_overall.get(f"ndcg@{k}", 0.0))
        overall_row_out[f"test/ndcg@{k}"] = float(test_overall.get(f"ndcg@{k}", 0.0))

    if val_warmup is not None and test_warmup is not None:
        val_click_w = metrics_row(val_warmup, "click")
        test_click_w = metrics_row(test_warmup, "click")
        val_purchase_w = metrics_row(val_warmup, "purchase")
        test_purchase_w = metrics_row(test_warmup, "purchase")
        val_overall_w = overall_row(val_warmup)
        test_overall_w = overall_row(test_warmup)

        click_row_w = {}
        purchase_row_w = {}
        for k in val_warmup["topk"]:
            click_row_w[f"val/hr@{k}"] = float(val_click_w.get(f"hr@{k}", 0.0))
            click_row_w[f"test/hr@{k}"] = float(test_click_w.get(f"hr@{k}", 0.0))
            click_row_w[f"val/ndcg@{k}"] = float(val_click_w.get(f"ndcg@{k}", 0.0))
            click_row_w[f"test/ndcg@{k}"] = float(test_click_w.get(f"ndcg@{k}", 0.0))

            purchase_row_w[f"val/hr@{k}"] = float(val_purchase_w.get(f"hr@{k}", 0.0))
            purchase_row_w[f"test/hr@{k}"] = float(test_purchase_w.get(f"hr@{k}", 0.0))
            purchase_row_w[f"val/ndcg@{k}"] = float(val_purchase_w.get(f"ndcg@{k}", 0.0))
            purchase_row_w[f"test/ndcg@{k}"] = float(test_purchase_w.get(f"ndcg@{k}", 0.0))

        overall_row_w_out = {}
        for k in val_warmup["topk"]:
            overall_row_w_out[f"val/ndcg@{k}"] = float(val_overall_w.get(f"ndcg@{k}", 0.0))
            overall_row_w_out[f"test/ndcg@{k}"] = float(test_overall_w.get(f"ndcg@{k}", 0.0))

        if not smoke_cpu:
            df_clicks_w = pd.DataFrame([click_row_w], index=["metrics"]).loc[:, col_order]
            df_purchase_w = pd.DataFrame([purchase_row_w], index=["metrics"]).loc[:, col_order]
            df_clicks_w.to_csv(run_dir / "results_clicks_warmup.csv", index=False)
            df_purchase_w.to_csv(run_dir / "results_purchase_warmup.csv", index=False)

            df_overall_w = pd.DataFrame([overall_row_w_out], index=["metrics"]).loc[:, overall_col_order]
            df_overall_w.to_csv(run_dir / "results_warmup.csv", index=False)

            with open(run_dir / "summary@10_warmup.txt", "w") as f:
                f.write(summary_at_k_text(val_warmup, test_warmup, k=10))

    if not smoke_cpu:
        df_overall = pd.DataFrame([overall_row_out], index=["metrics"]).loc[:, overall_col_order]
        df_overall.to_csv(run_dir / "results.csv", index=False)

        with open(run_dir / "summary@10.txt", "w") as f:
            f.write(summary_at_k_text(val_best, test_best, k=10))


__all__ = ["write_results"]

