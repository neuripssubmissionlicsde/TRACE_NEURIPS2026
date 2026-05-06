#!/usr/bin/env python3
"""
Multi-Seed Repetitions & Bootstrap Confidence Intervals (4.1 + 4.2)
====================================================================

Re-runs ONLY the generation phase (Phase 3) with N different seeds,
reusing the already-extracted factors and ImpactMatrix from final_run.

This isolates the stochastic variability of the SDE generator itself.
No LLM calls required ($0 cost).

Produces:
  - Per-seed quality metrics (KS, Wasserstein, ACF, Sharpe, etc.)
  - Cross-seed stability (mean ± std of all metrics)
  - Bootstrap 95% confidence intervals
  - MMD pairwise between synthetic samples from different seeds
  - JSON report + CSV summary + comparison plots

Usage:
    python scripts/eval_multiseed_bootstrap.py
    python scripts/eval_multiseed_bootstrap.py --n-seeds 5 --config configs/final_run.yaml
"""

from __future__ import annotations

import argparse
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


# ── Metric helpers ──────────────────────────────────────────────────


def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    result = np.correlate(x, x, mode="full")
    result = result[len(x) - 1:]
    return result[:nlags + 1] / (var * len(x))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    n = min(len(X), 5000)
    m = min(len(Y), 5000)
    if n < 5 or m < 5:
        return float("nan")
    rng = np.random.RandomState(0)
    X = X[rng.choice(len(X), n, replace=False)].reshape(-1, 1)
    Y = Y[rng.choice(len(Y), m, replace=False)].reshape(-1, 1)
    from scipy.spatial.distance import cdist
    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")
    Kxx = np.exp(-gamma * XX)
    Kyy = np.exp(-gamma * YY)
    Kxy = np.exp(-gamma * XY)
    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def compute_extended_metrics(real_rets: np.ndarray, synth_rets: np.ndarray) -> dict:
    base = compute_distributional_metrics(real_rets, synth_rets)
    base["wasserstein"] = float(stats.wasserstein_distance(real_rets, synth_rets))

    real_acf = acf(real_rets)
    synth_acf = acf(synth_rets)
    base["acf_dist_returns"] = float(np.mean(np.abs(real_acf - synth_acf)))

    real_abs_acf = acf(np.abs(real_rets))
    synth_abs_acf = acf(np.abs(synth_rets))
    base["acf_dist_abs_returns"] = float(np.mean(np.abs(real_abs_acf - synth_abs_acf)))

    window = min(21, len(real_rets) // 10)
    if window > 3:
        real_vol = pd.Series(real_rets).rolling(window).std().dropna().values
        synth_vol = pd.Series(synth_rets).rolling(window).std().dropna().values
        min_len = min(len(real_vol), len(synth_vol))
        if min_len > 5:
            base["vol_rmse"] = float(np.sqrt(np.mean(
                (real_vol[:min_len] - synth_vol[:min_len]) ** 2
            )))
            corr = np.corrcoef(real_vol[:min_len], synth_vol[:min_len])[0, 1]
            base["vol_corr"] = float(corr) if np.isfinite(corr) else 0.0

    real_cum = np.cumsum(real_rets)
    synth_cum = np.cumsum(synth_rets)
    min_len = min(len(real_cum), len(synth_cum))
    if min_len > 5:
        corr = np.corrcoef(real_cum[:min_len], synth_cum[:min_len])[0, 1]
        base["price_corr"] = float(corr) if np.isfinite(corr) else 0.0

    def _sharpe(r):
        if np.std(r) < 1e-10:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(252))

    def _max_dd(r):
        cum = np.cumsum(r)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    base["sharpe_diff"] = abs(_sharpe(real_rets) - _sharpe(synth_rets))
    base["max_dd_diff"] = abs(_max_dd(real_rets) - _max_dd(synth_rets))
    base["synth_mean"] = float(np.mean(synth_rets))
    base["synth_std"] = float(np.std(synth_rets))
    base["synth_skew"] = float(stats.skew(synth_rets))
    base["synth_kurt"] = float(stats.kurtosis(synth_rets, fisher=True))

    return base


# ── Bootstrap CI ────────────────────────────────────────────────────


def bootstrap_ci(values: list, n_bootstrap: int = 10000, alpha: float = 0.05) -> dict:
    """Compute bootstrap confidence interval for the mean."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) < 2:
        return {"mean": float(arr[0]) if len(arr) == 1 else None,
                "ci_lower": None, "ci_upper": None, "std": None, "n": len(arr)}
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(boot_means)),
        "n": len(arr),
    }


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-seed repetitions + Bootstrap CI")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "djia28.yaml"))
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of different seeds")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["AAPL"])
    output_name = cfg.get("output_dir", "final_run_gpt4o_mini")
    results_dir = PROJECT_ROOT / "results" / output_name
    n_samples = cfg.get("generation", {}).get("n_samples", 10)

    out_dir = PROJECT_ROOT / "results" / "multiseed_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["open", "high", "low", "close", "volume"]
    seeds = [args.base_seed + i * 7 for i in range(args.n_seeds)]

    print("=" * 70)
    print("  MULTI-SEED REPETITIONS + BOOTSTRAP CI")
    print("=" * 70)
    print(f"  Tickers:  {tickers}")
    print(f"  Seeds:    {seeds}")
    print(f"  Samples:  {n_samples} per seed")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    t_start = time.time()
    all_ticker_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        ticker_dir = results_dir / ticker

        # Load impact matrix
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ impact_matrix.json not found for {ticker}, skipping")
            continue
        im = ImpactMatrix.load(str(im_path))
        K = len(im.factor_names)
        print(f"  ✓ ImpactMatrix: {K} factors")

        # Load analyses
        analyses_path = ticker_dir / "analyses.json"
        if not analyses_path.exists():
            print(f"  ⚠ analyses.json not found for {ticker}, skipping")
            continue
        with open(analyses_path) as f:
            analyses = json.load(f)

        # Load real data
        training_csv = results_dir / "training_data.csv"
        if not training_csv.exists():
            print(f"  ⚠ training_data.csv not found, skipping")
            continue
        df = pd.read_csv(training_csv)
        df["date"] = pd.to_datetime(df["date"])
        ticker_df = df[df["tic"] == ticker].sort_values("date").copy()

        if len(ticker_df) < 100:
            print(f"  ⚠ insufficient data, skipping")
            continue

        feature_cols = [c for c in feature_names if c in ticker_df.columns]
        real_prices = ticker_df[feature_cols].values.astype(np.float64)
        real_prices = np.clip(real_prices, 1e-8, None)
        real_log_returns = np.diff(np.log(real_prices), axis=0)
        real_volume = ticker_df["volume"].values.astype(np.float64)
        real_dates = pd.to_datetime(ticker_df["date"]).values
        initial_prices = real_prices[0]
        n_steps = len(ticker_df)

        real_close = ticker_df["close"].values
        real_rets = np.diff(np.log(np.clip(real_close, 1e-8, None)))

        # Build historical presence
        dates = pd.to_datetime(ticker_df["date"]).values
        date_to_row = {d: i for i, d in enumerate(dates)}
        name_to_idx = {n: i for i, n in enumerate(im.factor_names)}
        full_presence = np.zeros((len(dates), K), dtype=np.float64)
        for w in analyses:
            ws = pd.Timestamp(w["start_date"])
            we = pd.Timestamp(w["end_date"])
            for f_dict in w.get("factors", []):
                col = name_to_idx.get(f_dict["name"])
                if col is None:
                    continue
                mag = abs(f_dict["magnitude"])
                for d in dates:
                    if ws <= pd.Timestamp(d) <= we:
                        row = date_to_row[d]
                        full_presence[row, col] = max(full_presence[row, col], mag)

        # Run generation with multiple seeds
        seed_metrics = []
        seed_returns = []

        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)

            gen = ImpactDrivenGenerator(im, window_trading_days=63)
            prices = gen.generate_scenario(
                n_steps=n_steps,
                initial_prices=initial_prices,
                n_samples=n_samples,
                noise_scale=1.0,
                seed=seed,
                real_log_returns=real_log_returns,
                real_volume=real_volume,
                historical_presence=full_presence,
                real_prices=real_prices,
                real_dates=real_dates,
            )

            close_col = min(3, prices.shape[2] - 1)
            synth_close = prices[0, :, close_col]
            synth_rets = np.diff(np.log(np.clip(synth_close, 1e-8, None)))

            metrics = compute_extended_metrics(real_rets, synth_rets)
            metrics["seed"] = seed

            # Average across all samples
            ks_all = []
            for s in range(n_samples):
                s_close = prices[s, :, close_col]
                s_rets = np.diff(np.log(np.clip(s_close, 1e-8, None)))
                ks, _ = stats.ks_2samp(real_rets, s_rets)
                ks_all.append(ks)
            metrics["mean_ks_all_samples"] = float(np.mean(ks_all))

            seed_metrics.append(metrics)
            seed_returns.append(synth_rets)
            print(f"KS={metrics['ks_statistic']:.4f} WD={metrics['wasserstein']:.5f}")

        # Cross-seed MMD
        cross_mmds = []
        for i in range(len(seed_returns)):
            for j in range(i + 1, len(seed_returns)):
                cross_mmds.append(mmd_rbf(seed_returns[i], seed_returns[j]))
        cross_mmds = [v for v in cross_mmds if np.isfinite(v)]

        # Bootstrap CI per metric
        metric_keys = [
            "ks_statistic", "wasserstein", "acf_dist_returns", "acf_dist_abs_returns",
            "sharpe_diff", "max_dd_diff", "vol_rmse", "vol_corr", "price_corr",
            "synth_mean", "synth_std", "synth_skew", "synth_kurt",
            "mean_ks_all_samples",
        ]
        bootstrap_results = {}
        for key in metric_keys:
            vals = [m.get(key) for m in seed_metrics]
            bootstrap_results[key] = bootstrap_ci(vals)

        # Print summary
        print(f"\n  {'Metric':<25s} {'Mean':>10s} {'Std':>10s} {'95% CI':>24s}")
        print(f"  {'─' * 72}")
        for key in metric_keys:
            b = bootstrap_results[key]
            if b["mean"] is not None:
                ci_str = f"[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]" if b["ci_lower"] is not None else "N/A"
                std_str = f"{b['std']:.4f}" if b["std"] is not None else "N/A"
                print(f"  {key:<25s} {b['mean']:>10.4f} {std_str:>10s} {ci_str:>24s}")

        if cross_mmds:
            print(f"\n  Cross-seed MMD: {np.mean(cross_mmds):.6f} ± {np.std(cross_mmds):.6f}")

        ticker_result = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_factors": K,
            "per_seed_metrics": seed_metrics,
            "bootstrap_ci": bootstrap_results,
            "cross_seed_mmd": {
                "mean": float(np.mean(cross_mmds)) if cross_mmds else None,
                "std": float(np.std(cross_mmds)) if cross_mmds else None,
                "values": cross_mmds,
            },
        }
        all_ticker_results[ticker] = ticker_result

    total_time = time.time() - t_start

    # Aggregate across tickers
    print(f"\n{'=' * 70}")
    print("  AGGREGATE BOOTSTRAP CI (across all tickers)")
    print(f"{'=' * 70}")

    agg_bootstrap = {}
    for key in metric_keys:
        all_means = []
        for tr in all_ticker_results.values():
            bc = tr.get("bootstrap_ci", {}).get(key, {})
            if bc.get("mean") is not None:
                all_means.append(bc["mean"])
        if all_means:
            agg_bootstrap[key] = bootstrap_ci(all_means)
            b = agg_bootstrap[key]
            ci_str = f"[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]" if b["ci_lower"] is not None else "N/A"
            std_str = f"{b['std']:.4f}" if b["std"] is not None else "N/A"
            print(f"  {key:<25s} {b['mean']:>10.4f} {std_str:>10s} {ci_str:>24s}")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")

    # Save report
    report = {
        "experiment": "multiseed_bootstrap",
        "description": "Multi-seed repetitions of generation phase + bootstrap 95% CI",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_bootstrap_samples": 10000,
        "confidence_level": 0.95,
        "tickers": list(all_ticker_results.keys()),
        "aggregate_bootstrap_ci": agg_bootstrap,
        "per_ticker": all_ticker_results,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = out_dir / "multiseed_bootstrap_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n  ✓ Report saved to {report_path}")

    # Per-seed CSV
    rows = []
    for ticker, tr in all_ticker_results.items():
        for m in tr.get("per_seed_metrics", []):
            row = {"ticker": ticker}
            row.update(m)
            rows.append(row)
    if rows:
        csv_path = out_dir / "per_seed_metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  ✓ Per-seed CSV: {csv_path}")

    # Bootstrap CI CSV
    ci_rows = []
    for ticker, tr in all_ticker_results.items():
        for key, bc in tr.get("bootstrap_ci", {}).items():
            row = {"ticker": ticker, "metric": key}
            row.update(bc)
            ci_rows.append(row)
    if ci_rows:
        ci_csv_path = out_dir / "bootstrap_ci.csv"
        pd.DataFrame(ci_rows).to_csv(ci_csv_path, index=False)
        print(f"  ✓ Bootstrap CI CSV: {ci_csv_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tickers_list = list(all_ticker_results.keys())
        if not tickers_list:
            return

        # 1. KS & Wasserstein with error bars across seeds
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, (key, title) in zip(axes, [
            ("ks_statistic", "KS Statistic"),
            ("wasserstein", "Wasserstein Distance"),
            ("sharpe_diff", "Sharpe Ratio Difference"),
        ]):
            means = []
            stds = []
            ci_lowers = []
            ci_uppers = []
            for t in tickers_list:
                bc = all_ticker_results[t]["bootstrap_ci"].get(key, {})
                m = bc.get("mean", 0)
                means.append(m)
                stds.append(bc.get("std", 0) or 0)
                ci_lowers.append(m - (bc.get("ci_lower", m) or m))
                ci_uppers.append((bc.get("ci_upper", m) or m) - m)

            x = np.arange(len(tickers_list))
            ax.bar(x, means, color="#1565C0", alpha=0.7, label="Mean")
            ax.errorbar(x, means, yerr=[ci_lowers, ci_uppers],
                       fmt="none", ecolor="black", capsize=4, capthick=1.5,
                       label="95% CI")
            ax.set_title(f"{title} (N={len(seeds)} seeds)", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(tickers_list)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Multi-Seed Stability with Bootstrap 95% CI", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plot_path = out_dir / "multiseed_bootstrap_comparison.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        fig.savefig(str(out_dir / "multiseed_bootstrap_comparison.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Comparison plot: {plot_path}")

        # 2. Box plot of per-seed metrics across tickers
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (key, title) in zip(axes, [
            ("ks_statistic", "KS Statistic"),
            ("wasserstein", "Wasserstein Distance"),
            ("sharpe_diff", "Sharpe Diff"),
        ]):
            data = []
            labels = []
            for t in tickers_list:
                vals = [m.get(key) for m in all_ticker_results[t]["per_seed_metrics"]
                        if m.get(key) is not None]
                if vals:
                    data.append(vals)
                    labels.append(t)
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("#1565C0")
                    patch.set_alpha(0.5)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Distribution Across {len(seeds)} Seeds", fontsize=14, fontweight="bold")
        fig.tight_layout()
        box_path = out_dir / "multiseed_boxplot.png"
        fig.savefig(str(box_path), dpi=150, bbox_inches="tight")
        fig.savefig(str(out_dir / "multiseed_boxplot.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Boxplot: {box_path}")

    except Exception as e:
        print(f"  ⚠ Plotting failed: {e}")


if __name__ == "__main__":
    main()
