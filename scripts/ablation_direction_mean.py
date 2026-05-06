#!/usr/bin/env python3
"""Bull/Bear stochastic dropout — mean synthetic close (N=30).

For each ticker, generates 30 synthetic trajectories under three
conditions (all factors, bears OFF, bulls OFF) and plots the mean
close price per condition.  This shows the *stochastic* effect of
factor removal on the generated price level, not just the analytical
drift.

Outputs:
    results/export_paper/direction_dropout_mean/
        dropout_mean_<TICKER>.{png,pdf}   (8 per-ticker plots)
        dropout_mean_summary.{png,pdf}    (2×4 grid)
        dropout_mean_report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    generate_synthetic,
    load_analyses,
)

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
N_SAMPLES = 30
SEED = 42
WINDOW_TRADING_DAYS = 63
RUN_DIR_NAME = "final_run_gpt4o_mini"


def _load_real(ticker: str, training_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def _bull_bear_masks(im: ImpactMatrix, eps: float = 1e-6):
    close_idx = min(3, im.base_impact.shape[1] - 1)
    impact_close = im.base_impact[:, close_idx]
    return impact_close > eps, impact_close < -eps


def _mean_close(synth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, mean_close) averaged over all samples."""
    dates = pd.to_datetime(
        synth_df[synth_df["sample"] == 0].sort_values("date")["date"]
    ).values
    n_steps = len(dates)
    closes = []
    for sid in sorted(synth_df["sample"].unique()):
        c = synth_df[synth_df["sample"] == sid].sort_values("date")["close"].values
        closes.append(c[:n_steps])
    return dates, np.mean(closes, axis=0)


def _plot_ticker(
    ticker: str,
    dates: np.ndarray,
    real_close: np.ndarray,
    mean_all: np.ndarray,
    mean_no_bears: np.ndarray,
    mean_no_bulls: np.ndarray,
    ax,
):
    ax.plot(dates, real_close, color="black", linewidth=1.4,
            label="Real", zorder=10)
    ax.plot(dates, mean_all, color="tab:blue", linewidth=1.2,
            label="All factors (N=30 mean)")
    ax.plot(dates, mean_no_bears, color="#ee6677", linewidth=1.2,
            linestyle="--", label="Bears OFF")
    ax.plot(dates, mean_no_bulls, color="#228833", linewidth=1.2,
            linestyle="--", label="Bulls OFF")
    ax.set_title(ticker)
    ax.set_ylabel("Close price (USD)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RUN_DIR_NAME
    out_dir = PROJECT_ROOT / "results" / "export_paper" / "direction_dropout_mean"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_csv = results_dir / "training_data.csv"

    report = {"n_samples": N_SAMPLES, "seed": SEED, "tickers": {}}

    summary_fig, summary_axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    summary_flat = summary_axes.flatten()

    t0 = time.time()
    for idx, ticker in enumerate(TICKERS):
        print(f"\n── {ticker} ──")
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

        bull_mask, bear_mask = _bull_bear_masks(im)
        n_bull, n_bear = int(bull_mask.sum()), int(bear_mask.sum())

        presence_no_bears = presence.copy()
        presence_no_bears[:, bear_mask] = 0.0
        presence_no_bulls = presence.copy()
        presence_no_bulls[:, bull_mask] = 0.0

        gen = ImpactDrivenGenerator(im, window_trading_days=WINDOW_TRADING_DAYS)

        # Generate under three conditions.
        print(f"  Generating all factors (N={N_SAMPLES})...")
        synth_all = generate_synthetic(
            gen, df_real, ticker, presence, N_SAMPLES, SEED)
        print(f"  Generating bears OFF...")
        synth_nb = generate_synthetic(
            gen, df_real, ticker, presence_no_bears, N_SAMPLES, SEED)
        print(f"  Generating bulls OFF...")
        synth_nbu = generate_synthetic(
            gen, df_real, ticker, presence_no_bulls, N_SAMPLES, SEED)

        dates_all, mean_all = _mean_close(synth_all)
        _, mean_nb = _mean_close(synth_nb)
        _, mean_nbu = _mean_close(synth_nbu)

        real_close = df_real["close"].astype(np.float64).values[:len(dates_all)]
        dates = dates_all

        # Terminal return (last / first - 1) for each condition.
        def _terminal(arr):
            return float((arr[-1] / arr[0] - 1) * 100)

        report["tickers"][ticker] = {
            "n_bull": n_bull, "n_bear": n_bear,
            "terminal_return_pct": {
                "real": _terminal(real_close),
                "all_factors": _terminal(mean_all),
                "bears_off": _terminal(mean_nb),
                "bulls_off": _terminal(mean_nbu),
            },
            "final_price": {
                "real": float(real_close[-1]),
                "all_factors": float(mean_all[-1]),
                "bears_off": float(mean_nb[-1]),
                "bulls_off": float(mean_nbu[-1]),
            },
        }

        # Per-ticker figure.
        fig, ax = plt.subplots(figsize=(11, 5))
        _plot_ticker(ticker, dates, real_close, mean_all, mean_nb, mean_nbu, ax)
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"dropout_mean_{ticker}.png", dpi=200)
        fig.savefig(out_dir / f"dropout_mean_{ticker}.pdf")
        plt.close(fig)

        # Summary subplot.
        _plot_ticker(ticker, dates, real_close, mean_all, mean_nb, mean_nbu, ax_sum)
        ax_sum.legend(fontsize=6.5, loc="best", frameon=False)

        tr = report["tickers"][ticker]["terminal_return_pct"]
        print(f"  ✓ terminal: real={tr['real']:+.1f}%  all={tr['all_factors']:+.1f}%  "
              f"bearsOff={tr['bears_off']:+.1f}%  bullsOff={tr['bulls_off']:+.1f}%")

    summary_fig.suptitle(
        f"Bull/Bear dropout — mean synthetic close (N={N_SAMPLES})", fontsize=14)
    summary_fig.tight_layout()
    summary_fig.savefig(out_dir / "dropout_mean_summary.png", dpi=180)
    summary_fig.savefig(out_dir / "dropout_mean_summary.pdf")
    plt.close(summary_fig)

    with open(out_dir / "dropout_mean_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n✓ Saved to {out_dir}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
