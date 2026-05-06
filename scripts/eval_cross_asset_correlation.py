#!/usr/bin/env python3
"""Item 6 — Cross-asset Pearson correlation heatmap (real vs LCSDE).

Computes 8x8 Pearson correlation matrices of daily log-returns for the
real and the LCSDE-synthetic data, plots them side-by-side, and writes
a JSON report with both matrices and the mean absolute error.

Outputs:
    results/final_run_gpt4o_mini/cross_asset_correlation/
        cross_asset_correlation_heatmap.png (and .pdf)
        cross_asset_correlation_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN", "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE", "PG", "TRV", "UNH", "V"]
RESULTS_DIR_NAME = "djia28"


def _log_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(np.float64).clip(lower=1e-8)
    return np.log(close).diff().dropna()


def _build_returns_matrix(close_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(close_by_ticker)[TICKERS]
    df = df.dropna(how="any")
    return np.log(df.clip(lower=1e-8)).diff().dropna()


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_dir = results_dir / "cross_asset_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real data via training_data.csv (matches what the model trained on).
    training_csv = results_dir / "training_data.csv"
    real_full = pd.read_csv(training_csv)
    real_full["date"] = pd.to_datetime(real_full["date"])

    real_close: dict[str, pd.Series] = {}
    synth_close: dict[str, pd.Series] = {}

    for ticker in TICKERS:
        rdf = (
            real_full[real_full["tic"] == ticker]
            .sort_values("date").set_index("date")
        )
        real_close[ticker] = rdf["close"].astype(np.float64)

        # Average of the 10 existing synthetic samples per ticker.
        ticker_dir = results_dir / ticker
        sample_files = sorted(ticker_dir.glob("synthetic_data_[0-9].csv"))
        if not sample_files:
            print(f"⚠ No synthetic CSVs for {ticker}, skipping.")
            continue

        sample_closes = []
        for sf in sample_files:
            sdf = pd.read_csv(sf)
            sdf["date"] = pd.to_datetime(sdf["date"])
            sdf = sdf.sort_values("date").set_index("date")
            sample_closes.append(sdf["close"].astype(np.float64))
        synth_avg = pd.concat(sample_closes, axis=1).mean(axis=1)
        synth_close[ticker] = synth_avg

    real_ret = _build_returns_matrix(real_close)
    synth_ret = _build_returns_matrix(synth_close)

    real_corr = real_ret.corr(method="pearson").loc[TICKERS, TICKERS]
    synth_corr = synth_ret.corr(method="pearson").loc[TICKERS, TICKERS]

    diff = (real_corr.values - synth_corr.values)
    mae = float(np.mean(np.abs(diff)))
    # Off-diagonal only (more informative, diag is always 1).
    mask_off = ~np.eye(len(TICKERS), dtype=bool)
    mae_off = float(np.mean(np.abs(diff[mask_off])))

    print(f"MAE (full)         = {mae:.4f}")
    print(f"MAE (off-diagonal) = {mae_off:.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = "RdBu_r"
    common_kwargs = dict(
        vmin=-1, vmax=1, cmap=cmap, annot=True, fmt=".2f",
        square=True, cbar=False, linewidths=0.5, linecolor="white",
        xticklabels=TICKERS, yticklabels=TICKERS,
        annot_kws={"size": 8},
    )

    sns.heatmap(real_corr, ax=axes[0], **common_kwargs)
    axes[0].set_title("Real")

    sns.heatmap(synth_corr, ax=axes[1], **common_kwargs)
    axes[1].set_title("LCSDE (avg of 10 samples)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Pearson correlation")

    fig.suptitle(
        f"Cross-asset log-return correlation — Real vs LCSDE  "
        f"(MAE off-diag = {mae_off:.3f})",
        y=1.02,
    )

    png = out_dir / "cross_asset_correlation_heatmap.png"
    pdf = out_dir / "cross_asset_correlation_heatmap.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = {
        "tickers": TICKERS,
        "real_corr": real_corr.values.tolist(),
        "synth_corr": synth_corr.values.tolist(),
        "abs_diff": np.abs(diff).tolist(),
        "mae": mae,
        "mae_off_diagonal": mae_off,
        "method": "pearson",
        "synth_aggregation": "mean of 10 sample CSVs",
    }
    json_path = out_dir / "cross_asset_correlation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved heatmap: {png}")
    print(f"✓ Saved report:  {json_path}")


if __name__ == "__main__":
    main()
