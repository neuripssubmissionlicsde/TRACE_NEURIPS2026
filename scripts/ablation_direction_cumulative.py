#!/usr/bin/env python3
"""Bull/Bear dropout — cumulative analytical signal per ticker.

Plots the cumulative deterministic drift signal under three conditions
(all factors, bears OFF, bulls OFF) for each ticker individually,
plus a 2×4 summary grid. Purely analytical — no stochastic generation.

Outputs:
    results/export_paper/direction_dropout_cumulative/
        dropout_cumulative_<TICKER>.{png,pdf}  (8 per-ticker)
        dropout_cumulative_summary.{png,pdf}   (2×4 grid)
"""

from __future__ import annotations

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
RUN_DIR_NAME = "final_run_gpt4o_mini"


def _load_real(ticker: str, training_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def _bull_bear_masks(im: ImpactMatrix, eps: float = 1e-6):
    close_idx = min(3, im.base_impact.shape[1] - 1)
    v = im.base_impact[:, close_idx]
    return v > eps, v < -eps


def _plot_ticker(ticker, dates, cum_all, cum_nb, cum_nbu, n_bull, n_bear, ax):
    ax.plot(dates, cum_all, color="tab:blue", linewidth=1.6,
            label="All factors")
    ax.plot(dates, cum_nb, color="#ee6677", linewidth=1.3,
            linestyle="--", label="Bears OFF")
    ax.plot(dates, cum_nbu, color="#228833", linewidth=1.3,
            linestyle="--", label="Bulls OFF")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

    # Annotate final values on the right edge.
    for val, color, ha in [
        (cum_all[-1], "tab:blue", "left"),
        (cum_nb[-1], "#ee6677", "left"),
        (cum_nbu[-1], "#228833", "left"),
    ]:
        ax.annotate(
            f"{val:+.2f}%", xy=(dates[-1], val),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7.5, color=color, va="center", ha=ha,
        )

    ax.set_title(f"{ticker}  (bull={n_bull}, bear={n_bear})")
    ax.set_ylabel("Cumulative drift (%)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RUN_DIR_NAME
    out_dir = PROJECT_ROOT / "results" / "export_paper" / "direction_dropout_cumulative"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_csv = results_dir / "training_data.csv"

    summary_fig, summary_axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    summary_flat = summary_axes.flatten()

    for idx, ticker in enumerate(TICKERS):
        ax_sum = summary_flat[idx]
        im_path = results_dir / ticker / "impact_matrix.json"
        if not im_path.exists():
            ax_sum.set_title(f"{ticker}\n(missing)")
            continue

        im = ImpactMatrix.load(str(im_path))
        analyses = load_analyses(ticker, results_dir)
        df_real = _load_real(ticker, training_csv)
        if df_real.empty:
            continue
        presence = build_presence_matrix(analyses, df_real, im.factor_names)
        daily_dates = pd.to_datetime(sorted(df_real["date"].unique())).values

        bull_mask, bear_mask = _bull_bear_masks(im)
        n_bull, n_bear = int(bull_mask.sum()), int(bear_mask.sum())

        presence_nb = presence.copy()
        presence_nb[:, bear_mask] = 0.0
        presence_nbu = presence.copy()
        presence_nbu[:, bull_mask] = 0.0

        sig_all = compute_analytical_signal(im, presence)
        sig_nb = compute_analytical_signal(im, presence_nb)
        sig_nbu = compute_analytical_signal(im, presence_nbu)

        cum_all = np.cumsum(sig_all) * 100   # → %
        cum_nb = np.cumsum(sig_nb) * 100
        cum_nbu = np.cumsum(sig_nbu) * 100

        # Per-ticker figure.
        fig, ax = plt.subplots(figsize=(11, 5))
        _plot_ticker(ticker, daily_dates, cum_all, cum_nb, cum_nbu,
                     n_bull, n_bear, ax)
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"dropout_cumulative_{ticker}.png", dpi=200)
        fig.savefig(out_dir / f"dropout_cumulative_{ticker}.pdf")
        plt.close(fig)

        # Summary subplot.
        _plot_ticker(ticker, daily_dates, cum_all, cum_nb, cum_nbu,
                     n_bull, n_bear, ax_sum)
        ax_sum.legend(fontsize=6.5, loc="best", frameon=False)

        print(f"  ✓ {ticker}: final cum — all={cum_all[-1]:+.2f}%  "
              f"bearsOff={cum_nb[-1]:+.2f}%  bullsOff={cum_nbu[-1]:+.2f}%")

    summary_fig.suptitle(
        "Bull/Bear dropout — cumulative analytical drift signal",
        fontsize=14,
    )
    summary_fig.tight_layout()
    summary_fig.savefig(out_dir / "dropout_cumulative_summary.png", dpi=180)
    summary_fig.savefig(out_dir / "dropout_cumulative_summary.pdf")
    plt.close(summary_fig)

    print(f"\n✓ Saved to {out_dir}")


if __name__ == "__main__":
    main()
