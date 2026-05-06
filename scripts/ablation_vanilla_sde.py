#!/usr/bin/env python3
"""
Ablation Study: Causal SDE vs Vanilla SDE (no causal factors)
==============================================================

Tests whether LLM-extracted causal factors actually improve the generated
data over an equivalent SDE that uses exactly the same stochastic engine
(GJR-GARCH, jump-diffusion, regime calibration) but with ZERO factors.

The Vanilla SDE keeps:
  - GJR-GARCH(1,1) conditional volatility
  - Skewed-t innovations
  - Merton jump-diffusion (regime-dependent)
  - Regime-switching drift calibration
  - OHLC microstructure model
  - Volume AR(1)

The Vanilla SDE removes:
  - All K causal factors (schedule, impacts, interactions, temporal
    profiles, factor noise)

Cost: $0 LLM calls — generates purely from calibrated stochastic model.

Usage:
    python scripts/ablation_vanilla_sde.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator
from sde_causal_generator.benchmark import compute_distributional_metrics


def make_vanilla_impact_matrix(feature_names: list[str]) -> ImpactMatrix:
    """Create an ImpactMatrix with zero factors (K=0)."""
    M = len(feature_names)
    return ImpactMatrix(
        factor_names=[],
        base_impact=np.zeros((0, M)),
        impact_std=np.zeros((0, M)),
        occurrence_prob=np.zeros(0),
        temporal_profile=np.zeros((0, 10)),
        interaction_matrix=np.zeros((0, 0)),
        feature_names=feature_names,
        nonlinearity_scores=np.zeros(0),
        response_curves=np.zeros((0, 3, M)),
    )


def main(config_path: str | None = None):
    import yaml

    path = config_path or str(PROJECT_ROOT / "configs" / "causal_sde_config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["AAPL"])
    output_name = cfg.get("output_dir", "multi_asset_2018_2023")
    results_dir = PROJECT_ROOT / "results" / output_name
    seed = cfg.get("seed", 42)
    n_samples = cfg.get("generation", {}).get("n_samples", 10)

    out_dir = PROJECT_ROOT / "results" / "ablation_vanilla_sde"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["open", "high", "low", "close", "volume"]

    print("═" * 70)
    print("  ABLATION: Causal SDE vs Vanilla SDE (no causal factors)")
    print("═" * 70)
    print(f"  Tickers:  {tickers}")
    print(f"  Samples:  {n_samples}")
    print(f"  Output:   {out_dir}")
    print("═" * 70)

    t_start = time.time()
    all_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        # ── Load real data ──────────────────────────────────────────
        training_csv = results_dir / "training_data.csv"
        df = pd.read_csv(training_csv)
        df["date"] = pd.to_datetime(df["date"])
        ticker_df = df[df["tic"] == ticker].sort_values("date").copy()

        if len(ticker_df) < 100:
            print(f"  ⚠ {ticker}: insufficient data, skipping")
            continue

        feature_cols = [c for c in feature_names if c in ticker_df.columns]
        real_prices = ticker_df[feature_cols].values.astype(np.float64)
        real_prices = np.clip(real_prices, 1e-8, None)
        real_log_returns = np.diff(np.log(real_prices), axis=0)
        real_volume = ticker_df["volume"].values.astype(np.float64)
        real_dates = pd.to_datetime(ticker_df["date"]).values
        initial_prices = real_prices[0]
        n_steps = len(ticker_df)

        # Close-only real returns for evaluation
        real_close = ticker_df.sort_values("date")["close"].values
        real_rets = np.diff(np.log(np.clip(real_close, 1e-8, None)))

        # ── Load Causal SDE results ─────────────────────────────────
        csde_csv = results_dir / ticker / "synthetic_data_causal_sde.csv"
        if csde_csv.exists():
            csde_df = pd.read_csv(csde_csv)
            if "sample" in csde_df.columns:
                csde_s0 = csde_df[csde_df["sample"] == 0].sort_values("date")
            else:
                csde_s0 = csde_df.sort_values("date")
            csde_close = csde_s0["close"].values
            csde_rets = np.diff(np.log(np.clip(csde_close, 1e-8, None)))
            print(f"  ✓ Causal SDE loaded ({len(csde_s0)} rows)")
        else:
            print(f"  ⚠ No Causal SDE results for {ticker}, skipping")
            continue

        # ── Generate Vanilla SDE data ───────────────────────────────
        print(f"  Generating Vanilla SDE (K=0 factors)...")
        np.random.seed(seed)

        vanilla_im = make_vanilla_impact_matrix(feature_cols)
        vanilla_gen = ImpactDrivenGenerator(vanilla_im, window_trading_days=63)

        vanilla_prices = vanilla_gen.generate_scenario(
            n_steps=n_steps,
            initial_prices=initial_prices,
            n_samples=n_samples,
            noise_scale=1.0,
            seed=seed,
            real_log_returns=real_log_returns,
            real_volume=real_volume,
            historical_presence=None,  # no factors
            real_prices=real_prices,
            real_dates=real_dates,
        )

        # Use sample 0 for comparison
        vanilla_close = vanilla_prices[0, :, min(3, vanilla_prices.shape[2] - 1)]
        vanilla_rets = np.diff(np.log(np.clip(vanilla_close, 1e-8, None)))
        print(f"  ✓ Vanilla SDE generated ({n_steps} steps × {n_samples} samples)")

        # ── Compute metrics for all 3 methods ──────────────────────
        csde_metrics = compute_distributional_metrics(real_rets, csde_rets)
        vanilla_metrics = compute_distributional_metrics(real_rets, vanilla_rets)

        # ACF distance (returns and absolute returns)
        def _acf(x, nlags=20):
            x = x - x.mean()
            var = np.var(x)
            if var < 1e-12:
                return np.zeros(nlags + 1)
            result = np.correlate(x, x, mode="full")
            result = result[len(x) - 1:]
            return result[:nlags + 1] / (var * len(x))

        real_acf = _acf(real_rets)
        csde_acf = _acf(csde_rets)
        vanilla_acf = _acf(vanilla_rets)
        csde_acf_dist = float(np.mean(np.abs(real_acf - csde_acf)))
        vanilla_acf_dist = float(np.mean(np.abs(real_acf - vanilla_acf)))

        # Absolute return ACF (volatility clustering)
        real_abs_acf = _acf(np.abs(real_rets))
        csde_abs_acf = _acf(np.abs(csde_rets))
        vanilla_abs_acf = _acf(np.abs(vanilla_rets))
        csde_abs_acf_dist = float(np.mean(np.abs(real_abs_acf - csde_abs_acf)))
        vanilla_abs_acf_dist = float(np.mean(np.abs(real_abs_acf - vanilla_abs_acf)))

        # Wasserstein distance
        from scipy.stats import wasserstein_distance
        csde_wass = float(wasserstein_distance(real_rets, csde_rets))
        vanilla_wass = float(wasserstein_distance(real_rets, vanilla_rets))

        # Multi-sample evaluation: average KS across all samples
        csde_ks_all = []
        vanilla_ks_all = []
        if csde_csv.exists() and "sample" in csde_df.columns:
            for s in range(min(n_samples, csde_df["sample"].max() + 1)):
                s_df = csde_df[csde_df["sample"] == s].sort_values("date")
                s_rets = np.diff(np.log(np.clip(s_df["close"].values, 1e-8, None)))
                ks, _ = stats.ks_2samp(real_rets, s_rets)
                csde_ks_all.append(ks)

        for s in range(n_samples):
            v_close = vanilla_prices[s, :, min(3, vanilla_prices.shape[2] - 1)]
            v_rets = np.diff(np.log(np.clip(v_close, 1e-8, None)))
            ks, _ = stats.ks_2samp(real_rets, v_rets)
            vanilla_ks_all.append(ks)

        ticker_results = {
            "causal_sde": {
                "ks_statistic": csde_metrics["ks_statistic"],
                "ks_pvalue": csde_metrics["ks_pvalue"],
                "volatility_ratio": csde_metrics["volatility_ratio"],
                "kurtosis_diff": csde_metrics["kurtosis_diff"],
                "skewness_diff": csde_metrics["skewness_diff"],
                "wasserstein": csde_wass,
                "acf_dist_returns": csde_acf_dist,
                "acf_dist_abs_returns": csde_abs_acf_dist,
                "mean_ks_all_samples": float(np.mean(csde_ks_all)) if csde_ks_all else csde_metrics["ks_statistic"],
            },
            "vanilla_sde": {
                "ks_statistic": vanilla_metrics["ks_statistic"],
                "ks_pvalue": vanilla_metrics["ks_pvalue"],
                "volatility_ratio": vanilla_metrics["volatility_ratio"],
                "kurtosis_diff": vanilla_metrics["kurtosis_diff"],
                "skewness_diff": vanilla_metrics["skewness_diff"],
                "wasserstein": vanilla_wass,
                "acf_dist_returns": vanilla_acf_dist,
                "acf_dist_abs_returns": vanilla_abs_acf_dist,
                "mean_ks_all_samples": float(np.mean(vanilla_ks_all)),
            },
        }
        all_results[ticker] = ticker_results

        # Print per-ticker summary
        print(f"\n  {'Metric':<25s} {'Causal SDE':>12s} {'Vanilla SDE':>12s} {'Winner':>10s}")
        print(f"  {'─' * 60}")
        for metric in ["ks_statistic", "wasserstein", "acf_dist_returns",
                        "acf_dist_abs_returns", "volatility_ratio",
                        "kurtosis_diff", "skewness_diff"]:
            cv = ticker_results["causal_sde"][metric]
            vv = ticker_results["vanilla_sde"][metric]
            if metric == "volatility_ratio":
                # Closer to 1.0 is better
                c_err = abs(cv - 1.0)
                v_err = abs(vv - 1.0)
                winner = "CSDE" if c_err <= v_err else "Vanilla"
            else:
                # Lower is better
                winner = "CSDE" if cv <= vv else "Vanilla"
            print(f"  {metric:<25s} {cv:>12.5f} {vv:>12.5f} {winner:>10s}")

    # ── Aggregate results ───────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'═' * 70}")
    print("  AGGREGATE COMPARISON")
    print(f"{'═' * 70}")

    metrics_to_compare = [
        "ks_statistic", "wasserstein", "acf_dist_returns",
        "acf_dist_abs_returns", "kurtosis_diff", "skewness_diff",
    ]

    csde_wins = 0
    vanilla_wins = 0
    total_comparisons = 0

    for metric in metrics_to_compare:
        c_vals = [all_results[t]["causal_sde"][metric] for t in all_results]
        v_vals = [all_results[t]["vanilla_sde"][metric] for t in all_results]
        c_mean = np.mean(c_vals)
        v_mean = np.mean(v_vals)

        c_win = sum(1 for c, v in zip(c_vals, v_vals) if c <= v)
        v_win = len(c_vals) - c_win
        csde_wins += c_win
        vanilla_wins += v_win
        total_comparisons += len(c_vals)

        winner = "CSDE" if c_mean <= v_mean else "Vanilla"
        print(f"  {metric:<25s}: CSDE={c_mean:.5f}  Vanilla={v_mean:.5f}  "
              f"Win: CSDE {c_win}/{len(c_vals)}  [{winner}]")

    # Volatility ratio (special: closer to 1.0)
    c_vr = [abs(all_results[t]["causal_sde"]["volatility_ratio"] - 1.0) for t in all_results]
    v_vr = [abs(all_results[t]["vanilla_sde"]["volatility_ratio"] - 1.0) for t in all_results]
    c_vr_mean = np.mean(c_vr)
    v_vr_mean = np.mean(v_vr)
    vr_c_win = sum(1 for c, v in zip(c_vr, v_vr) if c <= v)
    csde_wins += vr_c_win
    vanilla_wins += len(c_vr) - vr_c_win
    total_comparisons += len(c_vr)
    print(f"  {'vol_ratio_error':<25s}: CSDE={c_vr_mean:.5f}  Vanilla={v_vr_mean:.5f}  "
          f"Win: CSDE {vr_c_win}/{len(c_vr)}  [{'CSDE' if c_vr_mean <= v_vr_mean else 'Vanilla'}]")

    # Paired t-test on KS
    c_ks = [all_results[t]["causal_sde"]["ks_statistic"] for t in all_results]
    v_ks = [all_results[t]["vanilla_sde"]["ks_statistic"] for t in all_results]
    if len(c_ks) >= 3:
        t_stat, t_pval = stats.ttest_rel(c_ks, v_ks)
    else:
        t_stat, t_pval = 0.0, 1.0

    print(f"\n  Overall wins: CSDE {csde_wins}/{total_comparisons}  "
          f"Vanilla {vanilla_wins}/{total_comparisons}")
    print(f"  Paired t-test (KS): t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Total time: {total_time:.1f}s")

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "experiment": "ablation_vanilla_sde",
        "description": "Causal SDE (K factors) vs Vanilla SDE (K=0 factors, same GARCH+jumps+regime engine)",
        "tickers": list(all_results.keys()),
        "per_ticker": all_results,
        "aggregate": {
            "csde_wins": csde_wins,
            "vanilla_wins": vanilla_wins,
            "total_comparisons": total_comparisons,
            "paired_ttest_ks": {"t_statistic": float(t_stat), "p_value": float(t_pval)},
            "mean_csde_ks": float(np.mean(c_ks)),
            "mean_vanilla_ks": float(np.mean(v_ks)),
        },
        "total_time_seconds": round(total_time, 1),
    }

    report_path = out_dir / "vanilla_sde_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Report saved to {report_path}")

    # Generate comparison plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        tickers_list = list(all_results.keys())
        x = np.arange(len(tickers_list))
        width = 0.25

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # KS statistic
        ax = axes[0]
        ax.bar(x - width/2, [all_results[t]["causal_sde"]["ks_statistic"] for t in tickers_list],
               width, label="Causal SDE", color="#2196F3")
        ax.bar(x + width/2, [all_results[t]["vanilla_sde"]["ks_statistic"] for t in tickers_list],
               width, label="Vanilla SDE", color="#9E9E9E")
        ax.set_ylabel("KS Statistic")
        ax.set_title("KS Distance from Real (lower = better)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers_list)
        ax.legend()

        # Wasserstein
        ax = axes[1]
        ax.bar(x - width/2, [all_results[t]["causal_sde"]["wasserstein"] for t in tickers_list],
               width, label="Causal SDE", color="#2196F3")
        ax.bar(x + width/2, [all_results[t]["vanilla_sde"]["wasserstein"] for t in tickers_list],
               width, label="Vanilla SDE", color="#9E9E9E")
        ax.set_ylabel("Wasserstein Distance")
        ax.set_title("Wasserstein Distance (lower = better)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers_list)
        ax.legend()

        # ACF distance (absolute returns → vol clustering)
        ax = axes[2]
        ax.bar(x - width/2, [all_results[t]["causal_sde"]["acf_dist_abs_returns"] for t in tickers_list],
               width, label="Causal SDE", color="#2196F3")
        ax.bar(x + width/2, [all_results[t]["vanilla_sde"]["acf_dist_abs_returns"] for t in tickers_list],
               width, label="Vanilla SDE", color="#9E9E9E")
        ax.set_ylabel("ACF Distance (|returns|)")
        ax.set_title("Vol Clustering Fidelity (lower = better)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers_list)
        ax.legend()

        plt.tight_layout()
        fig.savefig(str(out_dir / "vanilla_sde_comparison.png"), dpi=150)
        with PdfPages(str(out_dir / "vanilla_sde_comparison.pdf")) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"  ✓ Plots saved")
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")

    print(f"\n{'═' * 70}")
    print("  ABLATION COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path=_path)
