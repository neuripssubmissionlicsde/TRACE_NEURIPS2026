#!/usr/bin/env python3
"""Item 5 — Re-run Phase 3 with N=30 trajectories.

Generates 30 synthetic trajectories per ticker using the already-trained
ImpactMatrix artefacts in ``results/final_run_gpt4o_mini/<TICKER>/``.
No LLM calls, no re-training of the impact matrix.

Outputs:
    results/final_run_gpt4o_mini/n30/<TICKER>/synthetic_data_{i}.csv
    results/final_run_gpt4o_mini/n30/real_vs_synth_AAPL_n30.png (and .pdf)
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    generate_synthetic,
    load_analyses,
)

TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN",
    "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE",
    "PG", "TRV", "UNH", "V",
]
N_SAMPLES = 30
SEED = 42
WINDOW_TRADING_DAYS = 63
RESULTS_DIR_NAME = "djia28"


def _load_real(ticker: str, training_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_root = results_dir / "n30"
    out_root.mkdir(parents=True, exist_ok=True)

    training_csv = results_dir / "training_data.csv"
    if not training_csv.exists():
        raise FileNotFoundError(f"Missing {training_csv}")

    aapl_real_close = None
    aapl_dates = None
    aapl_synth_closes: list[np.ndarray] = []

    for ticker in TICKERS:
        print(f"\n── {ticker} ──")
        ticker_dir = results_dir / ticker
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ Missing impact_matrix.json, skipping.")
            continue

        impact_matrix = ImpactMatrix.load(str(im_path))
        analyses = load_analyses(ticker, results_dir)
        df_real = _load_real(ticker, training_csv)
        if df_real.empty:
            print(f"  ⚠ No real data, skipping.")
            continue

        presence = build_presence_matrix(
            analyses, df_real, impact_matrix.factor_names
        )
        print(f"  Factors={len(impact_matrix.factor_names)}  Presence={presence.shape}")

        generator = ImpactDrivenGenerator(
            impact_matrix, window_trading_days=WINDOW_TRADING_DAYS
        )

        synth_df = generate_synthetic(
            generator, df_real, ticker, presence,
            n_samples=N_SAMPLES, seed=SEED,
        )

        # Save one CSV per sample (consistent with existing format).
        ticker_out = out_root / ticker
        ticker_out.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(synth_df["sample"].unique())
        for sid in sample_ids:
            sub = (
                synth_df[synth_df["sample"] == sid]
                .drop(columns=["sample"])
                [["open", "high", "low", "close", "volume", "date", "tic"]]
            )
            sub.to_csv(ticker_out / f"synthetic_data_{sid}.csv", index=False)

        if ticker == "AAPL":
            aapl_dates = pd.to_datetime(df_real["date"]).values
            aapl_real_close = df_real["close"].values.astype(np.float64)
            for sid in sample_ids:
                sub = synth_df[synth_df["sample"] == sid].sort_values("date")
                aapl_synth_closes.append(sub["close"].values.astype(np.float64))

        print(f"  ✓ Wrote {len(sample_ids)} CSVs to {ticker_out}")

    # ── AAPL envelope figure ──────────────────────────────────────
    if aapl_real_close is not None and aapl_synth_closes:
        synth_arr = np.array([
            s[: len(aapl_real_close)] for s in aapl_synth_closes
            if len(s) >= len(aapl_real_close)
        ])
        if synth_arr.size == 0:
            print("⚠ No AAPL synthetic data aligned for figure.")
            return

        synth_min = synth_arr.min(axis=0)
        synth_max = synth_arr.max(axis=0)
        synth_med = np.median(synth_arr, axis=0)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(
            aapl_dates, synth_min, synth_max,
            color="tab:blue", alpha=0.25,
            label=f"Synthetic envelope (N={N_SAMPLES})",
        )
        ax.plot(
            aapl_dates, synth_med,
            color="tab:blue", linewidth=1.0, alpha=0.85,
            label="Synthetic median",
        )
        ax.plot(
            aapl_dates, aapl_real_close,
            color="black", linewidth=1.3, label="Real AAPL close",
        )
        ax.set_title(f"AAPL — Real vs LCSDE synthetic envelope (N={N_SAMPLES})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close price (USD)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        png = out_root / "real_vs_synth_AAPL_n30.png"
        pdf = out_root / "real_vs_synth_AAPL_n30.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"\n✓ Saved figure: {png}")
        print(f"✓ Saved figure: {pdf}")


if __name__ == "__main__":
    main()
