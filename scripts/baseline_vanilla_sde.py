#!/usr/bin/env python3
"""Generate Vanilla SDE (K=0) per-ticker synthetic CSVs.

This complements ``scripts/ablation_vanilla_sde.py`` (which only
computes metrics in memory) by persisting the per-ticker Vanilla SDE
trajectories to CSV files compatible with the LCSDE format.

Outputs:
    results/final_run_gpt4o_mini/ablation_vanilla_sde/<TICKER>/
        synthetic_data_{0..N-1}.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
N_SAMPLES = 10
SEED = 42
RESULTS_DIR_NAME = "final_run_gpt4o_mini"
FEATURE_NAMES = ["open", "high", "low", "close", "volume"]


def _vanilla_impact_matrix() -> ImpactMatrix:
    M = len(FEATURE_NAMES)
    return ImpactMatrix(
        factor_names=[],
        base_impact=np.zeros((0, M)),
        impact_std=np.zeros((0, M)),
        occurrence_prob=np.zeros(0),
        temporal_profile=np.zeros((0, 10)),
        interaction_matrix=np.zeros((0, 0)),
        feature_names=FEATURE_NAMES,
        nonlinearity_scores=np.zeros(0),
        response_curves=np.zeros((0, 3, M)),
    )


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_root = results_dir / "ablation_vanilla_sde"
    out_root.mkdir(parents=True, exist_ok=True)

    training_csv = results_dir / "training_data.csv"
    real_full = pd.read_csv(training_csv)
    real_full["date"] = pd.to_datetime(real_full["date"])

    vanilla_im = _vanilla_impact_matrix()

    for ticker in TICKERS:
        print(f"\n── {ticker} (vanilla) ──")
        ticker_df = (
            real_full[real_full["tic"] == ticker]
            .sort_values("date").reset_index(drop=True)
        )
        if len(ticker_df) < 100:
            print(f"  ⚠ insufficient real data, skipping")
            continue

        feature_cols = [c for c in FEATURE_NAMES if c in ticker_df.columns]
        real_prices = ticker_df[feature_cols].values.astype(np.float64)
        real_prices = np.clip(real_prices, 1e-8, None)
        real_log_returns = np.diff(np.log(real_prices), axis=0)
        real_volume = ticker_df["volume"].values.astype(np.float64)
        real_dates = pd.to_datetime(ticker_df["date"]).values
        initial_prices = real_prices[0]
        n_steps = len(ticker_df)

        np.random.seed(SEED)
        gen = ImpactDrivenGenerator(vanilla_im, window_trading_days=63)

        prices = gen.generate_scenario(
            n_steps=n_steps,
            initial_prices=initial_prices,
            n_samples=N_SAMPLES,
            noise_scale=1.0,
            seed=SEED,
            real_log_returns=real_log_returns,
            real_volume=real_volume,
            historical_presence=None,   # K=0
            real_prices=real_prices,
            real_dates=real_dates,
        )  # (N_SAMPLES, n_steps, M)

        ticker_out = out_root / ticker
        ticker_out.mkdir(parents=True, exist_ok=True)

        date_strs = pd.to_datetime(real_dates).strftime("%Y-%m-%d").values
        cols = vanilla_im.feature_names[: prices.shape[2]]
        for s in range(prices.shape[0]):
            sdf = pd.DataFrame(prices[s], columns=cols)
            sdf["date"] = date_strs
            sdf["tic"] = ticker
            sdf = sdf[["open", "high", "low", "close", "volume", "date", "tic"]]
            sdf.to_csv(ticker_out / f"synthetic_data_{s}.csv", index=False)

        print(f"  ✓ wrote {prices.shape[0]} CSVs to {ticker_out}")


if __name__ == "__main__":
    main()
