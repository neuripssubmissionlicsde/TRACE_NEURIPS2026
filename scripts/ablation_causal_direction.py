#!/usr/bin/env python3
"""
Causal Direction Ablation — Bull vs Bear Factor Effect on Mean Return

For each ticker, loads the already-trained impact matrix and reconstructed
presence matrix from the completed pipeline run.  Measures the effect of
selectively disabling bull or bear factors on the **factor-driven drift
signal** — the deterministic component that the impact matrix injects
into the SDE before stochastic noise and regime calibration.

Three conditions:
  1. ALL factors active             (baseline)
  2. Bears OFF — only bull factors  (bear columns zeroed)
  3. Bulls OFF — only bear factors  (bull columns zeroed)

Reports:
  • Analytical factor signal (deterministic drift contribution)
  • Generated mean log-return (stochastic, 10 samples, seed=42)
  • Per-ticker and aggregate results

Expected behaviour:
  • Bears off  → signal/mean increases  (negative pressure removed)
  • Bulls off  → signal/mean decreases  (positive pressure removed)

Usage:
    python scripts/ablation_causal_direction.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import (
    CausalFactor,
    ImpactMatrix,
    WindowAnalysis,
)
from sde_causal_generator.generate import ImpactDrivenGenerator

# ── Configuration (defaults, overridable via CLI) ────────────────────
DEFAULT_RESULTS_DIR = "djia28"
TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN",
    "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE",
    "PG", "TRV", "UNH", "V",
]
SEED = 42
N_SAMPLES = 10
WINDOW_TRADING_DAYS = 63


def load_analyses(ticker: str, results_dir: Path = None) -> list[WindowAnalysis]:
    """Load cached WindowAnalysis objects from analyses.json."""
    path = (results_dir or (PROJECT_ROOT / "results" / DEFAULT_RESULTS_DIR)) / ticker / "analyses.json"
    with open(path) as f:
        raw = json.load(f)
    return [WindowAnalysis.from_dict(d) for d in raw]


def build_presence_matrix(
    analyses: list[WindowAnalysis],
    df: pd.DataFrame,
    factor_names: list[str],
    threshold: float = 0.0,
) -> np.ndarray:
    """Reconstruct presence matrix aligned to a given factor_names list."""
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    dates = pd.to_datetime(sorted(df["date"].unique()))
    T = len(dates)
    K = len(factor_names)
    presence = np.zeros((T, K), dtype=np.float32)
    date_to_row = {d: i for i, d in enumerate(dates)}

    for analysis in analyses:
        start = pd.Timestamp(analysis.start_date)
        end = pd.Timestamp(analysis.end_date)
        for f in analysis.factors:
            if f.magnitude <= threshold or f.name not in name_to_idx:
                continue
            col = name_to_idx[f.name]
            for d in dates:
                if start <= d <= end:
                    row = date_to_row[d]
                    presence[row, col] = max(presence[row, col], f.magnitude)
    return presence


def compute_llm_directions(
    analyses: list[WindowAnalysis],
    factor_names: list[str],
) -> np.ndarray:
    """Average LLM direction per factor. Positive=bull, negative=bear."""
    K = len(factor_names)
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    sums = np.zeros(K, dtype=np.float64)
    counts = np.zeros(K, dtype=np.float64)
    for analysis in analyses:
        for f in analysis.factors:
            if f.name in name_to_idx:
                idx = name_to_idx[f.name]
                sums[idx] += f.direction
                counts[idx] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        directions = np.where(counts > 0, sums / counts, 0.0)
    return directions.astype(np.float32)


def compute_analytical_signal(
    impact_matrix: ImpactMatrix,
    presence: np.ndarray,
) -> np.ndarray:
    """Compute the deterministic factor-driven drift signal.

    signal(t) = Σ_k  presence(t,k) × base_impact(k, close) × (1/window_days)

    This is the EXACT drift contribution injected by the impact matrix
    into the SDE before stochastic noise, GARCH, jumps, and calibration.

    Returns: (T,) daily drift signal.
    """
    close_idx = min(3, impact_matrix.base_impact.shape[1] - 1)
    daily_scale = 1.0 / WINDOW_TRADING_DAYS
    # (K,) — daily impact per factor on close
    daily_impact = impact_matrix.base_impact[:, close_idx] * daily_scale
    # (T,) — sum across factors
    signal = presence @ daily_impact
    return signal


def generate_synthetic(
    generator: ImpactDrivenGenerator,
    df: pd.DataFrame,
    ticker: str,
    presence: np.ndarray,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Run the ImpactDrivenGenerator with the given presence matrix."""
    np.random.seed(seed)
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df_sorted = df.sort_values("date")
    first_row = df_sorted.iloc[0]
    initial_prices = np.array(
        [first_row.get(c, 100.0) for c in feature_cols], dtype=np.float32
    )
    real_prices = df_sorted[feature_cols].values.astype(np.float64)
    real_prices = np.clip(real_prices, 1e-8, None)
    real_log_returns = np.diff(np.log(real_prices), axis=0)
    real_volume = df_sorted["volume"].values.astype(np.float64) if "volume" in df_sorted.columns else None
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
    )
    return synth_df


def compute_mean_return(synth_df: pd.DataFrame) -> float:
    """Mean daily log-return of synthetic close prices (across all samples)."""
    log_rets = []
    for sample_id in synth_df["sample"].unique():
        mask = synth_df["sample"] == sample_id
        c = synth_df.loc[mask, "close"].values
        lr = np.diff(np.log(np.clip(c, 1e-8, None)))
        log_rets.append(lr)
    all_lr = np.concatenate(log_rets)
    return float(np.mean(all_lr))


def main(results_dir_name: str | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Causal Direction Ablation")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory name (e.g. final_run_gpt4o_mini)")
    args = parser.parse_args()

    rd_name = results_dir_name or args.results_dir or DEFAULT_RESULTS_DIR
    RESULTS_DIR = PROJECT_ROOT / "results" / rd_name
    TRAINING_DATA = RESULTS_DIR / "training_data.csv"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "ablation_causal_direction"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # Load training data
    train_df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded training data: {len(train_df)} rows, "
          f"tickers: {train_df['tic'].unique().tolist()}")

    # If the results dir contains tickers beyond the default 8, use them.
    available_tickers = sorted(train_df["tic"].unique().tolist())
    tickers_iter = available_tickers if len(available_tickers) > len(TICKERS) else TICKERS

    results = {}

    for ticker in tickers_iter:
        print(f"\n{'═' * 60}")
        print(f"  {ticker}")
        print(f"{'═' * 60}")

        # Ticker's real data
        df_ticker = train_df[train_df["tic"] == ticker].copy()
        if len(df_ticker) == 0:
            print(f"  ⚠ No data for {ticker}, skipping.")
            continue

        # Load impact matrix (already trained)
        im_path = RESULTS_DIR / ticker / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ No impact_matrix.json for {ticker}, skipping.")
            continue
        impact_matrix = ImpactMatrix.load(str(im_path))
        factor_names = impact_matrix.factor_names
        K = len(factor_names)

        # Load analyses and reconstruct presence matrix
        analyses = load_analyses(ticker, RESULTS_DIR)
        presence = build_presence_matrix(analyses, df_ticker, factor_names)
        print(f"  Factors: {K}, Presence shape: {presence.shape}")

        # Determine factor direction from learned impact (close column)
        # This is what actually drives the generated drift.
        close_idx = min(3, impact_matrix.base_impact.shape[1] - 1)
        impact_close = impact_matrix.base_impact[:, close_idx]

        # Also get LLM directions for reference
        llm_dirs = compute_llm_directions(analyses, factor_names)

        # Classify factors by their FIN-learned close impact direction
        bull_mask = impact_close > 1e-6    # positive impact on close
        bear_mask = impact_close < -1e-6   # negative impact on close
        neutral_mask = ~bull_mask & ~bear_mask

        n_bull = int(bull_mask.sum())
        n_bear = int(bear_mask.sum())
        n_neutral = int(neutral_mask.sum())
        print(f"  Direction (FIN impact): {n_bull} bull, {n_bear} bear, {n_neutral} neutral")

        # Check FIN-LLM agreement
        llm_bull = llm_dirs > 0.05
        llm_bear = llm_dirs < -0.05
        agree = ((bull_mask & llm_bull) | (bear_mask & llm_bear) | (neutral_mask & ~llm_bull & ~llm_bear)).sum()
        print(f"  FIN-LLM direction agreement: {agree}/{K} ({100*agree/K:.0f}%)")

        # Build factor lists for report
        bull_factors = [factor_names[i] for i in range(K) if bull_mask[i]]
        bear_factors = [factor_names[i] for i in range(K) if bear_mask[i]]

        # ── Analytical Signal (deterministic, no noise) ──────────────
        print(f"\n  ─── Analytical Factor Signal (deterministic) ───")

        signal_all = compute_analytical_signal(impact_matrix, presence)

        presence_no_bears = presence.copy()
        presence_no_bears[:, bear_mask] = 0.0
        signal_no_bears = compute_analytical_signal(impact_matrix, presence_no_bears)

        presence_no_bulls = presence.copy()
        presence_no_bulls[:, bull_mask] = 0.0
        signal_no_bulls = compute_analytical_signal(impact_matrix, presence_no_bulls)

        # Cumulative drift and mean daily drift
        cum_all = float(signal_all.sum())
        cum_no_bears = float(signal_no_bears.sum())
        cum_no_bulls = float(signal_no_bulls.sum())
        mean_all = float(signal_all.mean())
        mean_no_bears = float(signal_no_bears.mean())
        mean_no_bulls = float(signal_no_bulls.mean())

        print(f"  All factors:  mean_daily={mean_all:+.6f}  cumul={cum_all:+.4f}")
        print(f"  Bears OFF:    mean_daily={mean_no_bears:+.6f}  cumul={cum_no_bears:+.4f}")
        print(f"  Bulls OFF:    mean_daily={mean_no_bulls:+.6f}  cumul={cum_no_bulls:+.4f}")
        print(f"  Δ(bears_off): {mean_no_bears - mean_all:+.6f}  "
              f"{'✓ UP' if mean_no_bears > mean_all else '✗ DOWN'}")
        print(f"  Δ(bulls_off): {mean_no_bulls - mean_all:+.6f}  "
              f"{'✓ DOWN' if mean_no_bulls < mean_all else '✗ UP'}")

        # ── Generated Data (stochastic, with regime calibration) ─────
        print(f"\n  ─── Generated Data (stochastic, n_samples={N_SAMPLES}) ───")
        generator = ImpactDrivenGenerator(impact_matrix, window_trading_days=WINDOW_TRADING_DAYS)

        conditions = {
            "all_factors": presence,
            "bears_off": presence_no_bears,
            "bulls_off": presence_no_bulls,
        }
        gen_results = {}
        for cond_name, cond_presence in conditions.items():
            print(f"  Generating: {cond_name} ...", end=" ", flush=True)
            synth_df = generate_synthetic(
                generator, df_ticker, ticker, cond_presence, N_SAMPLES, SEED,
            )
            mean_ret = compute_mean_return(synth_df)
            ann_mean = mean_ret * 252

            sample_means = []
            for sid in synth_df["sample"].unique():
                c = synth_df.loc[synth_df["sample"] == sid, "close"].values
                lr = np.diff(np.log(np.clip(c, 1e-8, None)))
                sample_means.append(float(np.mean(lr)))

            gen_results[cond_name] = {
                "mean_daily_log_return": mean_ret,
                "annualised_mean": ann_mean,
                "per_sample_means": sample_means,
                "std_across_samples": float(np.std(sample_means)),
            }
            print(f"mean_lr={mean_ret:.6f}  (ann={ann_mean:.4f})")

        # ── Compile ticker result ────────────────────────────────────
        ticker_result = {
            "n_factors": K,
            "n_bull": n_bull,
            "n_bear": n_bear,
            "n_neutral": int(neutral_mask.sum()),
            "bull_factors": bull_factors,
            "bear_factors": bear_factors,
            "analytical_signal": {
                "all_factors": {
                    "mean_daily": mean_all,
                    "cumulative": cum_all,
                    "annualised": mean_all * 252,
                },
                "bears_off": {
                    "mean_daily": mean_no_bears,
                    "cumulative": cum_no_bears,
                    "annualised": mean_no_bears * 252,
                },
                "bulls_off": {
                    "mean_daily": mean_no_bulls,
                    "cumulative": cum_no_bulls,
                    "annualised": mean_no_bulls * 252,
                },
                "delta_bears_off": mean_no_bears - mean_all,
                "delta_bulls_off": mean_no_bulls - mean_all,
                "bears_off_increases_signal": mean_no_bears > mean_all,
                "bulls_off_decreases_signal": mean_no_bulls < mean_all,
            },
            "generated": gen_results,
            "generated_delta_bears_off": (
                gen_results["bears_off"]["mean_daily_log_return"]
                - gen_results["all_factors"]["mean_daily_log_return"]
            ),
            "generated_delta_bulls_off": (
                gen_results["bulls_off"]["mean_daily_log_return"]
                - gen_results["all_factors"]["mean_daily_log_return"]
            ),
        }
        results[ticker] = ticker_result

    # ── Aggregate ────────────────────────────────────────────────────
    n_total = len(results)
    n_sig_bears_up = sum(
        1 for r in results.values()
        if r["analytical_signal"]["bears_off_increases_signal"]
    )
    n_sig_bulls_down = sum(
        1 for r in results.values()
        if r["analytical_signal"]["bulls_off_decreases_signal"]
    )

    mean_delta_bears_sig = float(np.mean([
        r["analytical_signal"]["delta_bears_off"] for r in results.values()
    ]))
    mean_delta_bulls_sig = float(np.mean([
        r["analytical_signal"]["delta_bulls_off"] for r in results.values()
    ]))

    # Annualised signal means per condition
    ann_all = float(np.mean([
        r["analytical_signal"]["all_factors"]["annualised"] for r in results.values()
    ]))
    ann_bears_off = float(np.mean([
        r["analytical_signal"]["bears_off"]["annualised"] for r in results.values()
    ]))
    ann_bulls_off = float(np.mean([
        r["analytical_signal"]["bulls_off"]["annualised"] for r in results.values()
    ]))

    aggregate = {
        "n_tickers": n_total,
        "analytical_signal": {
            "bears_off_increases_signal": n_sig_bears_up,
            "bulls_off_decreases_signal": n_sig_bulls_down,
            "bears_off_rate": n_sig_bears_up / n_total if n_total else 0,
            "bulls_off_rate": n_sig_bulls_down / n_total if n_total else 0,
            "mean_delta_bears_off": mean_delta_bears_sig,
            "mean_delta_bulls_off": mean_delta_bulls_sig,
            "annualised_all": ann_all,
            "annualised_bears_off": ann_bears_off,
            "annualised_bulls_off": ann_bulls_off,
        },
    }

    total_time = time.time() - t_start

    report = {
        "experiment": "causal_direction_ablation",
        "description": (
            "Three conditions: all factors, bears OFF (only bulls+neutral), "
            "bulls OFF (only bears+neutral). Measures: (a) the deterministic "
            "factor signal (drift contribution from impact matrix × presence), "
            "and (b) the generated mean log-return. The analytical signal "
            "isolates the causal mechanism before stochastic noise and "
            "regime calibration."
        ),
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "window_trading_days": WINDOW_TRADING_DAYS,
        "tickers": tickers_iter,
        "per_ticker": results,
        "aggregate": aggregate,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = OUTPUT_DIR / "causal_direction_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY — Analytical Factor Signal")
    print(f"{'═' * 60}")
    print(f"  Bears OFF → signal increases: {n_sig_bears_up}/{n_total}")
    print(f"  Bulls OFF → signal decreases: {n_sig_bulls_down}/{n_total}")
    print(f"  Mean Δ(bears_off): {mean_delta_bears_sig:+.8f}")
    print(f"  Mean Δ(bulls_off): {mean_delta_bulls_sig:+.8f}")
    print(f"  Annualised signal: all={ann_all:+.4f}  bears_off={ann_bears_off:+.4f}  bulls_off={ann_bulls_off:+.4f}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Report: {report_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
