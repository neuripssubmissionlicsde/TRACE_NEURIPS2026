#!/usr/bin/env python3
"""Bull/Bear counterfactual dropout — mean synthetic close (N=30).

Uses ``counterfactual=True`` to disable regime drift calibration and
continuous drift correction, so the factor-driven signal actually
affects the generated price trajectory.

For each ticker, generates 30 trajectories under three conditions:
    1. All factors active
    2. Bear factors zeroed
    3. Bull factors zeroed

Outputs:
    results/export_paper/direction_dropout_counterfactual/
        dropout_cf_<TICKER>.{png,pdf}      (8 per-ticker)
        dropout_cf_summary.{png,pdf}       (2×4 grid)
        dropout_cf_report.json
"""

from __future__ import annotations

import json
import sys
import time
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
from sde_causal_generator.generate import ImpactDrivenGenerator
from scripts.ablation_causal_direction import (
    build_presence_matrix,
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
    v = im.base_impact[:, close_idx]
    return v > eps, v < -eps


def _generate_cf(
    generator: ImpactDrivenGenerator,
    df_real: pd.DataFrame,
    ticker: str,
    presence: np.ndarray,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Run generator in counterfactual mode."""
    np.random.seed(seed)
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df_real.columns]
    df_sorted = df_real.sort_values("date")
    first_row = df_sorted.iloc[0]
    initial_prices = np.array(
        [first_row.get(c, 100.0) for c in feature_cols], dtype=np.float32)
    real_prices = df_sorted[feature_cols].values.astype(np.float64)
    real_prices = np.clip(real_prices, 1e-8, None)
    real_log_returns = np.diff(np.log(real_prices), axis=0)
    real_volume = df_sorted["volume"].values.astype(np.float64)
    real_dates_arr = pd.to_datetime(df_sorted["date"]).values
    n_steps = len(real_dates_arr)

    synth_df = generator.generate_finrl_df(
        ticker=ticker,
        n_steps=n_steps,
        initial_prices=initial_prices,
        n_samples=n_samples,
        real_dates=real_dates_arr,
        start_date=str(real_dates_arr[0])[:10],
        real_log_returns=real_log_returns,
        real_volume=real_volume,
        historical_presence=presence,
        real_prices=real_prices,
        counterfactual=True,
    )
    return synth_df


def _mean_close(synth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(
        synth_df[synth_df["sample"] == 0].sort_values("date")["date"]
    ).values
    n = len(dates)
    closes = []
    for sid in sorted(synth_df["sample"].unique()):
        c = synth_df[synth_df["sample"] == sid].sort_values("date")["close"].values
        closes.append(c[:n])
    return dates, np.mean(closes, axis=0)


def _plot_ticker(ticker, dates, real_close, mean_all, mean_nb, mean_nbu, ax):
    ax.plot(dates, real_close, color="black", linewidth=1.4,
            label="Real", zorder=10)
    ax.plot(dates, mean_all, color="tab:blue", linewidth=1.2,
            label="All factors")
    ax.plot(dates, mean_nb, color="#ee6677", linewidth=1.2,
            linestyle="--", label="Bears OFF")
    ax.plot(dates, mean_nbu, color="#228833", linewidth=1.2,
            linestyle="--", label="Bulls OFF")
    ax.set_title(ticker)
    ax.set_ylabel("Close price (USD)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RUN_DIR_NAME
    out_dir = (
        PROJECT_ROOT / "results" / "export_paper"
        / "direction_dropout_counterfactual"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    training_csv = results_dir / "training_data.csv"

    report = {"n_samples": N_SAMPLES, "seed": SEED, "mode": "counterfactual",
              "tickers": {}}

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

        presence_nb = presence.copy()
        presence_nb[:, bear_mask] = 0.0
        presence_nbu = presence.copy()
        presence_nbu[:, bull_mask] = 0.0

        gen = ImpactDrivenGenerator(im, window_trading_days=WINDOW_TRADING_DAYS)

        # Use the SAME random state for all 3 conditions so that the
        # only difference is the factor presence (not the noise draw).
        print(f"  All factors (N={N_SAMPLES})...")
        rng_state = np.random.RandomState(SEED).get_state()
        np.random.set_state(rng_state)
        synth_all = _generate_cf(gen, df_real, ticker, presence, N_SAMPLES, SEED)

        print(f"  Bears OFF...")
        np.random.set_state(rng_state)
        synth_nb = _generate_cf(gen, df_real, ticker, presence_nb, N_SAMPLES, SEED)

        print(f"  Bulls OFF...")
        np.random.set_state(rng_state)
        synth_nbu = _generate_cf(gen, df_real, ticker, presence_nbu, N_SAMPLES, SEED)

        dates, mean_all = _mean_close(synth_all)
        _, mean_nb = _mean_close(synth_nb)
        _, mean_nbu = _mean_close(synth_nbu)

        real_close = df_real["close"].astype(np.float64).values[:len(dates)]

        def _terminal(a): return float((a[-1] / a[0] - 1) * 100)

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

        fig, ax = plt.subplots(figsize=(11, 5))
        _plot_ticker(ticker, dates, real_close, mean_all, mean_nb, mean_nbu, ax)
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"dropout_cf_{ticker}.png", dpi=200)
        fig.savefig(out_dir / f"dropout_cf_{ticker}.pdf")
        plt.close(fig)

        _plot_ticker(ticker, dates, real_close, mean_all, mean_nb, mean_nbu, ax_sum)
        ax_sum.legend(fontsize=6.5, loc="best", frameon=False)

        tr = report["tickers"][ticker]["terminal_return_pct"]
        print(f"  ✓ terminal: real={tr['real']:+.1f}%  all={tr['all_factors']:+.1f}%  "
              f"bearsOff={tr['bears_off']:+.1f}%  bullsOff={tr['bulls_off']:+.1f}%")

    summary_fig.suptitle(
        f"Bull/Bear counterfactual dropout — mean synthetic close "
        f"(N={N_SAMPLES}, drift calibration OFF)", fontsize=13)
    summary_fig.tight_layout()
    summary_fig.savefig(out_dir / "dropout_cf_summary.png", dpi=180)
    summary_fig.savefig(out_dir / "dropout_cf_summary.pdf")
    plt.close(summary_fig)

    with open(out_dir / "dropout_cf_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Saved to {out_dir}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
