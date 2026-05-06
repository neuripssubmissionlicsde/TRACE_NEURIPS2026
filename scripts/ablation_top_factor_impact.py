#!/usr/bin/env python3
"""Item 8 — Top-K analytical dropout per ticker.

For each ticker, identifies the top-5 factors by mean |base_impact|
(across features), then computes the deterministic drift signal under
6 conditions: all factors active, and each top-5 factor zeroed
individually. Plots the per-window time series of the annualised signal.

No stochastic generation is performed — only the analytical drift.

Outputs:
    results/final_run_gpt4o_mini/factor_dropout/
        top_factor_dropout_report.json
        top_factor_dropout_<TICKER>.png   (one per ticker)
        top_factor_dropout_summary.png    (2x4 grid)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    compute_analytical_signal,
    load_analyses,
)

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
TOP_K = 5
RESULTS_DIR_NAME = "final_run_gpt4o_mini"
TRADING_DAYS_PER_YEAR = 252


def _aggregate_per_window(
    daily_signal: np.ndarray,
    daily_dates: np.ndarray,
    analyses,
):
    """Mean of the daily analytical signal over each window's date range."""
    daily_dates = pd.to_datetime(daily_dates)
    rows = []
    sorted_analyses = sorted(analyses, key=lambda a: pd.Timestamp(a.start_date))
    for a in sorted_analyses:
        start = pd.Timestamp(a.start_date)
        end = pd.Timestamp(a.end_date)
        mask = (daily_dates >= start) & (daily_dates <= end)
        if not mask.any():
            continue
        rows.append((start + (end - start) / 2, float(daily_signal[mask].mean())))
    if not rows:
        return np.array([]), np.array([])
    dates, values = zip(*rows)
    return np.array(dates), np.array(values)


def _load_real(ticker: str, training_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def _plot_ticker(
    ticker: str,
    window_dates: np.ndarray,
    daily_dates: np.ndarray,
    window_series: dict[str, np.ndarray],
    daily_series: dict[str, np.ndarray],
    axes,
):
    """Two-panel plot: (top) annualised drift per window, (bottom) cumulative
    factor contribution to log-return (%)."""
    ax_top, ax_bot = axes
    palette = plt.cm.tab10.colors

    # ── Top: annualised drift per window ─────────────────────────
    all_y = window_series["all_factors"] * TRADING_DAYS_PER_YEAR * 100
    ax_top.plot(
        window_dates, all_y, color="black", linewidth=1.8,
        label="All factors", zorder=10,
    )
    color_idx = 0
    for label, sig in window_series.items():
        if label == "all_factors":
            continue
        ax_top.plot(
            window_dates, sig * TRADING_DAYS_PER_YEAR * 100,
            linewidth=1.0, alpha=0.85,
            color=palette[color_idx % len(palette)],
            label=f"− {label}",
        )
        color_idx += 1
    ax_top.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax_top.set_title(f"{ticker} — annualised drift per window")
    ax_top.set_ylabel("Annualised drift (%)")
    ax_top.grid(alpha=0.3)
    ax_top.xaxis.set_major_locator(mdates.YearLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Bottom: cumulative deterministic factor contribution ─────
    cum_all = np.cumsum(daily_series["all_factors"]) * 100  # %
    ax_bot.plot(
        daily_dates, cum_all, color="black", linewidth=1.8,
        label="All factors", zorder=10,
    )
    color_idx = 0
    for label, sig in daily_series.items():
        if label == "all_factors":
            continue
        cum = np.cumsum(sig) * 100
        ax_bot.plot(
            daily_dates, cum, linewidth=1.1, alpha=0.9,
            color=palette[color_idx % len(palette)],
            label=f"− {label}",
        )
        color_idx += 1
    ax_bot.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax_bot.set_title(f"{ticker} — cumulative factor contribution to log-return")
    ax_bot.set_ylabel("Cumulative drift (%)")
    ax_bot.set_xlabel("Date")
    ax_bot.grid(alpha=0.3)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_dir = results_dir / "factor_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_csv = results_dir / "training_data.csv"
    report = {"tickers": {}, "top_k": TOP_K}

    summary_fig, summary_axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    summary_axes_flat = summary_axes.flatten()

    for idx, ticker in enumerate(TICKERS):
        ax_summary = summary_axes_flat[idx]
        ticker_dir = results_dir / ticker
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            ax_summary.set_title(f"{ticker}\n(missing impact matrix)")
            continue

        im = ImpactMatrix.load(str(im_path))
        analyses = load_analyses(ticker, results_dir)
        df_real = _load_real(ticker, training_csv)
        if df_real.empty:
            continue
        presence = build_presence_matrix(analyses, df_real, im.factor_names)
        daily_dates = pd.to_datetime(sorted(df_real["date"].unique())).values

        # Top-K factors by mean |base_impact| across features.
        magnitudes = np.abs(im.base_impact).mean(axis=1)
        order = np.argsort(-magnitudes)
        top_idx = [int(i) for i in order[:TOP_K] if magnitudes[i] > 0]
        top_factors = [im.factor_names[i] for i in top_idx]

        # All-factors daily signal.
        signal_all = compute_analytical_signal(im, presence)
        win_dates, win_all = _aggregate_per_window(
            signal_all, daily_dates, analyses
        )
        daily_series = {"all_factors": signal_all}
        series_window = {"all_factors": win_all}
        per_factor_summary = {}
        for k_idx, fname in zip(top_idx, top_factors):
            presence_drop = presence.copy()
            presence_drop[:, k_idx] = 0.0
            sig_drop = compute_analytical_signal(im, presence_drop)
            _, win_drop = _aggregate_per_window(
                sig_drop, daily_dates, analyses
            )
            series_window[fname] = win_drop
            daily_series[fname] = sig_drop
            cum_with = float(np.sum(signal_all)) * 100  # %
            cum_without = float(np.sum(sig_drop)) * 100
            per_factor_summary[fname] = {
                "magnitude_mean_abs_impact": float(magnitudes[k_idx]),
                "annualised_signal_with": float(
                    win_all.mean() * TRADING_DAYS_PER_YEAR
                ),
                "annualised_signal_without": float(
                    win_drop.mean() * TRADING_DAYS_PER_YEAR
                ),
                "annualised_signal_delta": float(
                    (win_drop.mean() - win_all.mean()) * TRADING_DAYS_PER_YEAR
                ),
                "cumulative_drift_pct_with": cum_with,
                "cumulative_drift_pct_without": cum_without,
                "cumulative_drift_pct_delta": cum_without - cum_with,
            }

        # ── Per-ticker figure (2 panels) ────────────────────────
        fig, axes_pair = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        _plot_ticker(
            ticker, win_dates, daily_dates,
            series_window, daily_series, axes_pair,
        )
        axes_pair[0].legend(fontsize=8, loc="best", ncol=2, frameon=False)
        axes_pair[1].legend(fontsize=8, loc="best", ncol=2, frameon=False)
        fig.tight_layout()
        out_png = out_dir / f"top_factor_dropout_{ticker}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        # Summary subplot — cumulative view (most informative across tickers).
        cum_all = np.cumsum(signal_all) * 100
        ax_summary.plot(
            daily_dates, cum_all, color="black", linewidth=1.8,
            label="All factors", zorder=10,
        )
        palette = plt.cm.tab10.colors
        for c_i, (fname, sig) in enumerate(
            (k, v) for k, v in daily_series.items() if k != "all_factors"
        ):
            ax_summary.plot(
                daily_dates, np.cumsum(sig) * 100,
                linewidth=1.0, alpha=0.9,
                color=palette[c_i % len(palette)],
                label=f"− {fname}",
            )
        ax_summary.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax_summary.set_title(ticker)
        ax_summary.set_ylabel("Cumulative drift (%)")
        ax_summary.grid(alpha=0.3)
        ax_summary.xaxis.set_major_locator(mdates.YearLocator())
        ax_summary.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_summary.legend(fontsize=6, loc="best", ncol=1, frameon=False)

        report["tickers"][ticker] = {
            "top_factors": top_factors,
            "annualised_signal_all": float(
                win_all.mean() * TRADING_DAYS_PER_YEAR
            ),
            "cumulative_drift_pct_all": float(np.sum(signal_all)) * 100,
            "per_factor": per_factor_summary,
        }

        print(f"  ✓ {ticker} — top-{TOP_K}: {top_factors[:3]}...")

    summary_fig.suptitle(
        f"Top-{TOP_K} factor dropout — cumulative deterministic drift contribution",
        fontsize=14,
    )
    summary_fig.tight_layout()
    summary_png = out_dir / "top_factor_dropout_summary.png"
    summary_fig.savefig(summary_png, dpi=180)
    plt.close(summary_fig)

    with open(out_dir / "top_factor_dropout_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Saved per-ticker figures + summary in {out_dir}")


if __name__ == "__main__":
    main()
