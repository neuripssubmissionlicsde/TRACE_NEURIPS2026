#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility Benchmark — Multi-Run Concordance & Stability Analysis.

Runs the Causal SDE pipeline N times with different seeds and measures:

A. Factor extraction stability:
   - Jaccard similarity of factor sets across runs
   - Spearman rank correlation of factor magnitudes
   - Fleiss' Kappa for direction (bullish/bearish) agreement

B. Synthetic data quality (per run):
   - KS statistic, Wasserstein distance, moments
   - MMD (Maximum Mean Discrepancy) with RBF kernel
   - Signature distance (path-level comparison)

C. Cross-run synthetic stability:
   - Pairwise MMD between synthetic samples from different runs
   - Moment stability (std of moments across runs)

D. Output:
   - JSON report with all metrics
   - Summary tables (console + CSV)
   - Recommended protocol for publication

Usage
-----
    python scripts/generation/benchmark_reproducibility.py \\
        --config configs/causal_sde_config.yaml \\
        --n-runs 10 \\
        --output-dir results/reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices[1:] / prices[:-1])
    return np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    A kernel-based two-sample test statistic.  Lower = more similar.
    """
    n = min(len(X), 5000)  # subsample for efficiency
    m = min(len(Y), 5000)
    if n < 5 or m < 5:
        return float("nan")

    rng = np.random.RandomState(42)
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


def signature_distance(
    X: np.ndarray, Y: np.ndarray, depth: int = 4
) -> float:
    """
    Signature distance between two 1-D time series.

    Uses the ``signatory`` or ``iisignature`` package if available;
    falls back to a truncated path-moment heuristic.
    """
    try:
        import iisignature  # type: ignore[import-untyped]

        # Embed as 2-D path: (time, value)
        def _make_path(arr):
            t = np.linspace(0, 1, len(arr)).reshape(-1, 1)
            v = arr.reshape(-1, 1)
            return np.hstack([t, v]).astype(np.float64)

        sig_x = iisignature.sig(_make_path(X), depth)
        sig_y = iisignature.sig(_make_path(Y), depth)
        return float(np.linalg.norm(sig_x - sig_y))
    except ImportError:
        pass

    # Fallback: compare rolling statistics as path proxy
    window = min(20, len(X) // 3, len(Y) // 3)
    if window < 3:
        return float("nan")

    def _rolling_stats(arr, w):
        s = pd.Series(arr)
        return np.concatenate([
            s.rolling(w, min_periods=1).mean().values,
            s.rolling(w, min_periods=1).std().fillna(0).values,
            s.rolling(w, min_periods=1).skew().fillna(0).values,
        ])

    sx = _rolling_stats(X, window)
    sy = _rolling_stats(Y, window)
    n = min(len(sx), len(sy))
    return float(np.linalg.norm(sx[:n] - sy[:n]) / max(n, 1))


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for inter-rater agreement.

    Parameters
    ----------
    ratings_matrix : ndarray, shape (n_subjects, n_categories)
        Each row sums to the number of raters.  ratings_matrix[i, j]
        is the number of raters who assigned subject i to category j.
    """
    N, k = ratings_matrix.shape
    n = ratings_matrix[0].sum()  # number of raters per subject
    if n <= 1 or N == 0:
        return float("nan")

    # Proportion per category
    p = ratings_matrix.sum(axis=0) / (N * n)
    # Per-subject agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p ** 2)

    if abs(1.0 - P_e) < 1e-10:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


# ══════════════════════════════════════════════════════════════════════
# Single run
# ══════════════════════════════════════════════════════════════════════


def _run_pipeline_once(
    config_path: str,
    seed: int,
    run_id: int,
    base_output_dir: str,
) -> Dict[str, Any]:
    """Execute one pipeline run and collect artefacts."""
    from sde_causal_generator.pipeline import (
        CausalSDEPipeline,
        PipelineConfig,
    )

    cfg = PipelineConfig.from_yaml(config_path)
    cfg.seed = seed
    run_dir = os.path.join(base_output_dir, f"run_{run_id:02d}")
    cfg.output_dir = run_dir
    # Use a per-run cache for LLM factors so each run re-extracts
    cfg.cache_dir = os.path.join(run_dir, "cache")

    os.makedirs(run_dir, exist_ok=True)

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Load data
    import yaml
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f) or {}

    tickers = raw_cfg.get("tickers", ["AAPL"])
    train_start = raw_cfg.get("train_start", "2004-01-01")
    train_end = raw_cfg.get("train_end", "2020-06-30")
    test_start = raw_cfg.get("test_start", "2020-07-01")
    test_end = raw_cfg.get("test_end", "2021-12-31")

    ticker = tickers[0]

    pipe = CausalSDEPipeline(config=cfg)

    # Download data
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=train_start, end=train_end, progress=False)
        if hasattr(raw.columns, 'droplevel'):
            raw.columns = raw.columns.droplevel(1) if raw.columns.nlevels > 1 else raw.columns
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "adj close" in df.columns:
            df = df.drop(columns=["adj close"])

        test_raw = yf.download(ticker, start=test_start, end=test_end, progress=False)
        if hasattr(test_raw.columns, 'droplevel'):
            test_raw.columns = test_raw.columns.droplevel(1) if test_raw.columns.nlevels > 1 else test_raw.columns
        test_df = test_raw.reset_index()
        test_df.columns = [c.lower() for c in test_df.columns]
        if "adj close" in test_df.columns:
            test_df = test_df.drop(columns=["adj close"])
    except Exception as e:
        print(f"  [Run {run_id}] Failed to download data: {e}")
        return {"run_id": run_id, "error": str(e)}

    # Disable interactive mode for benchmark
    cfg.interactive_post_extraction = False
    cfg.interactive_pre_generation = False

    print(f"\n{'='*60}")
    print(f"  REPRODUCIBILITY RUN {run_id} (seed={seed})")
    print(f"{'='*60}")

    t0 = time.time()
    report = pipe.run(df=df, ticker=ticker, test_df=test_df)
    elapsed = time.time() - t0

    # Collect artefacts for concordance analysis
    result: Dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "report": report,
        "output_dir": run_dir,
    }

    # Load factor names and analyses
    analyses_path = os.path.join(run_dir, "analyses.json")
    if os.path.exists(analyses_path):
        with open(analyses_path) as f:
            analyses = json.load(f)
        factor_set = set()
        factor_magnitudes: Dict[str, float] = {}
        factor_directions: Dict[str, float] = {}
        for a in analyses:
            for fac in a.get("factors", []):
                name = fac.get("name", "")
                factor_set.add(name)
                factor_magnitudes[name] = factor_magnitudes.get(name, 0) + fac.get("magnitude", 0)
                factor_directions[name] = fac.get("direction", 0)
        result["factor_set"] = factor_set
        result["factor_magnitudes"] = factor_magnitudes
        result["factor_directions"] = factor_directions

    # Load synthetic data
    synth_path = os.path.join(run_dir, "synthetic_data.csv")
    if os.path.exists(synth_path):
        synth_df = pd.read_csv(synth_path)
        if "sample" in synth_df.columns:
            s0 = synth_df[synth_df["sample"] == 0].sort_values("date")
        else:
            s0 = synth_df.sort_values("date")
        result["synth_returns"] = _log_returns(s0["close"].values.astype(float))
    else:
        result["synth_returns"] = np.array([])

    return result


# ══════════════════════════════════════════════════════════════════════
# Concordance analysis
# ══════════════════════════════════════════════════════════════════════


def analyse_concordance(
    results: List[Dict[str, Any]],
    train_returns: np.ndarray,
) -> Dict[str, Any]:
    """Compute concordance and stability metrics across runs."""
    report: Dict[str, Any] = {}
    valid = [r for r in results if "error" not in r]

    if len(valid) < 2:
        report["error"] = "Need at least 2 successful runs"
        return report

    N = len(valid)
    report["n_runs"] = N

    # ── A. Factor concordance ───────────────────────────────────────
    factor_sets = [r.get("factor_set", set()) for r in valid]
    pairwise_jaccard = []
    for i in range(N):
        for j in range(i + 1, N):
            pairwise_jaccard.append(jaccard_similarity(factor_sets[i], factor_sets[j]))

    report["factor_concordance"] = {
        "mean_jaccard": float(np.mean(pairwise_jaccard)),
        "std_jaccard": float(np.std(pairwise_jaccard)),
        "min_jaccard": float(np.min(pairwise_jaccard)),
        "max_jaccard": float(np.max(pairwise_jaccard)),
    }

    # Spearman rank correlation of magnitudes
    all_factors = set()
    for fs in factor_sets:
        all_factors |= fs

    if len(all_factors) >= 3:
        mag_matrix = np.zeros((N, len(all_factors)))
        factor_list = sorted(all_factors)
        for i, r in enumerate(valid):
            mags = r.get("factor_magnitudes", {})
            for j, fn in enumerate(factor_list):
                mag_matrix[i, j] = mags.get(fn, 0)

        spearman_corrs = []
        for i in range(N):
            for j in range(i + 1, N):
                corr, _ = stats.spearmanr(mag_matrix[i], mag_matrix[j])
                if np.isfinite(corr):
                    spearman_corrs.append(corr)

        if spearman_corrs:
            report["factor_concordance"]["mean_spearman"] = float(np.mean(spearman_corrs))
            report["factor_concordance"]["std_spearman"] = float(np.std(spearman_corrs))

    # Fleiss' Kappa for direction agreement
    dir_ratings = []  # (n_factors, 3) — bullish, neutral, bearish counts
    for fn in sorted(all_factors):
        counts = [0, 0, 0]  # bullish, neutral, bearish
        for r in valid:
            dirs = r.get("factor_directions", {})
            d = dirs.get(fn, 0)
            if d > 0.1:
                counts[0] += 1  # bullish
            elif d < -0.1:
                counts[2] += 1  # bearish
            else:
                counts[1] += 1  # neutral
        dir_ratings.append(counts)

    if dir_ratings:
        ratings_matrix = np.array(dir_ratings)
        kappa = fleiss_kappa(ratings_matrix)
        report["factor_concordance"]["fleiss_kappa_direction"] = float(kappa)

    # ── B. Per-run quality metrics ──────────────────────────────────
    per_run_metrics = []
    for r in valid:
        synth_ret = r.get("synth_returns", np.array([]))
        if len(synth_ret) < 10 or len(train_returns) < 10:
            continue

        ks_stat, ks_pval = stats.ks_2samp(synth_ret, train_returns)
        wd = stats.wasserstein_distance(synth_ret, train_returns)
        mmd_val = mmd_rbf(synth_ret, train_returns)
        sig_val = signature_distance(synth_ret[:1000], train_returns[:1000])

        m = {
            "run_id": r["run_id"],
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "wasserstein": float(wd),
            "mmd_rbf": float(mmd_val) if np.isfinite(mmd_val) else None,
            "signature_dist": float(sig_val) if np.isfinite(sig_val) else None,
            "synth_mean": float(np.mean(synth_ret)),
            "synth_std": float(np.std(synth_ret)),
            "synth_skew": float(stats.skew(synth_ret)),
            "synth_kurt": float(stats.kurtosis(synth_ret, fisher=True)),
        }

        # OOS metrics from pipeline report
        oos = r.get("report", {}).get("phases", {}).get("oos_validation", {}).get("metrics", {})
        if oos:
            m["oos_ks"] = oos.get("oos_ks_stat")
            m["oos_wasserstein"] = oos.get("oos_wasserstein")

        per_run_metrics.append(m)

    report["per_run_metrics"] = per_run_metrics

    # ── C. Cross-run stability ──────────────────────────────────────
    if len(per_run_metrics) >= 2:
        stability = {}
        for key in ["ks_stat", "wasserstein", "mmd_rbf", "synth_mean",
                     "synth_std", "synth_skew", "synth_kurt"]:
            vals = [m[key] for m in per_run_metrics if m.get(key) is not None]
            if vals:
                stability[f"{key}_mean"] = float(np.mean(vals))
                stability[f"{key}_std"] = float(np.std(vals))
                stability[f"{key}_cv"] = float(np.std(vals) / max(abs(np.mean(vals)), 1e-10))

        # Pairwise MMD between synthetic samples from different runs
        synth_series = [r.get("synth_returns", np.array([])) for r in valid]
        synth_series = [s for s in synth_series if len(s) > 10]
        if len(synth_series) >= 2:
            cross_mmds = []
            for i in range(len(synth_series)):
                for j in range(i + 1, len(synth_series)):
                    cross_mmds.append(mmd_rbf(synth_series[i], synth_series[j]))
            cross_mmds = [v for v in cross_mmds if np.isfinite(v)]
            if cross_mmds:
                stability["cross_run_mmd_mean"] = float(np.mean(cross_mmds))
                stability["cross_run_mmd_std"] = float(np.std(cross_mmds))

        report["stability"] = stability

    return report


def print_concordance_report(report: Dict[str, Any]) -> None:
    """Print a formatted concordance report."""
    print(f"\n{'═' * 60}")
    print("  REPRODUCIBILITY BENCHMARK REPORT")
    print(f"{'═' * 60}")
    print(f"  Runs completed: {report.get('n_runs', 0)}")

    fc = report.get("factor_concordance", {})
    if fc:
        print(f"\n  ── Factor Concordance ──")
        print(f"    Jaccard similarity:    {fc.get('mean_jaccard', 0):.3f} ± {fc.get('std_jaccard', 0):.3f}")
        if "mean_spearman" in fc:
            print(f"    Spearman rank corr:    {fc['mean_spearman']:.3f} ± {fc.get('std_spearman', 0):.3f}")
        if "fleiss_kappa_direction" in fc:
            kappa = fc["fleiss_kappa_direction"]
            quality = (
                "almost perfect" if kappa > 0.81 else
                "substantial" if kappa > 0.61 else
                "moderate" if kappa > 0.41 else
                "fair" if kappa > 0.21 else
                "slight" if kappa > 0.0 else
                "poor"
            )
            print(f"    Fleiss' κ (direction): {kappa:.3f} ({quality})")

    stab = report.get("stability", {})
    if stab:
        print(f"\n  ── Synthetic Data Stability ──")
        for key in ["ks_stat", "wasserstein", "mmd_rbf", "synth_std",
                     "synth_skew", "synth_kurt"]:
            mean_k = f"{key}_mean"
            std_k = f"{key}_std"
            if mean_k in stab:
                label = key.replace("_", " ").title()
                print(f"    {label:<25s} {stab[mean_k]:>10.6f} ± {stab.get(std_k, 0):.6f}")

        if "cross_run_mmd_mean" in stab:
            print(f"    {'Cross-run MMD':<25s} {stab['cross_run_mmd_mean']:>10.6f} ± {stab.get('cross_run_mmd_std', 0):.6f}")

    prm = report.get("per_run_metrics", [])
    if prm:
        print(f"\n  ── Per-Run Metrics ──")
        header = f"    {'Run':>4s}  {'KS':>8s}  {'WD':>8s}  {'MMD':>8s}  {'Sig':>8s}  {'Std':>8s}  {'Kurt':>8s}"
        print(header)
        print(f"    {'─' * len(header)}")
        for m in prm:
            mmd_s = f"{m['mmd_rbf']:.4f}" if m.get("mmd_rbf") else "   N/A"
            sig_s = f"{m['signature_dist']:.4f}" if m.get("signature_dist") else "   N/A"
            print(
                f"    {m['run_id']:>4d}  {m['ks_stat']:>8.4f}  "
                f"{m['wasserstein']:>8.5f}  {mmd_s:>8s}  {sig_s:>8s}  "
                f"{m['synth_std']:>8.5f}  {m['synth_kurt']:>8.2f}"
            )

    print(f"\n{'═' * 60}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility benchmark for Causal SDE pipeline."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to causal_sde_config.yaml",
    )
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of independent runs (default: 10)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base seed.  Run i uses seed = base_seed + i (default: 42)",
    )
    parser.add_argument(
        "--output-dir", default="results/reproducibility",
        help="Directory for all outputs (default: results/reproducibility)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"  Config:    {args.config}")
    print(f"  Runs:      {args.n_runs}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Output:    {args.output_dir}")

    # Download training data once for baseline returns
    import yaml
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f) or {}

    ticker = raw_cfg.get("tickers", ["AAPL"])[0]
    train_start = raw_cfg.get("train_start", "2004-01-01")
    train_end = raw_cfg.get("train_end", "2020-06-30")

    try:
        import yfinance as yf
        raw = yf.download(ticker, start=train_start, end=train_end, progress=False)
        if hasattr(raw.columns, 'droplevel'):
            raw.columns = raw.columns.droplevel(1) if raw.columns.nlevels > 1 else raw.columns
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        train_returns = _log_returns(df["close"].values.astype(float))
    except Exception:
        train_returns = np.array([])

    # Run pipeline N times
    results: List[Dict[str, Any]] = []
    for i in range(args.n_runs):
        seed = args.base_seed + i
        result = _run_pipeline_once(
            config_path=args.config,
            seed=seed,
            run_id=i,
            base_output_dir=args.output_dir,
        )
        results.append(result)

    # Concordance analysis
    concordance = analyse_concordance(results, train_returns)
    print_concordance_report(concordance)

    # Save report
    report_path = os.path.join(args.output_dir, "reproducibility_report.json")

    # Make report JSON-serializable
    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    with open(report_path, "w") as f:
        json.dump(concordance, f, indent=2, default=_serialise)

    print(f"\n  Report saved to: {report_path}")

    # Save per-run summary CSV
    prm = concordance.get("per_run_metrics", [])
    if prm:
        csv_path = os.path.join(args.output_dir, "per_run_metrics.csv")
        pd.DataFrame(prm).to_csv(csv_path, index=False)
        print(f"  Per-run CSV:     {csv_path}")


if __name__ == "__main__":
    main()
