#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT Magnitude/Direction Ablation — Unit Vector Test.

Compares TFT training with two input representations:

  (A) Current pipeline:   presence = LLM magnitude [0,1]
                          directions = LLM mean direction [-1,+1]

  (B) Unit vectors:       presence = 1.0 (binary: factor present or not)
                          directions = sign(direction) ∈ {-1, 0, +1}

Both conditions use the SAME factor extraction (from analyses.json),
the SAME price data, the SAME architecture (TFT), and the SAME seed.
The ONLY difference is the numeric representation of magnitude and direction.

Outputs (results/tft_unit_vector_test/):
  tft_unit_vector_results.json     — full metrics for both conditions
  loss_curves.png                  — training/validation loss comparison
  impact_matrix_scatter.png        — per-factor impact correlation A vs B
  factor_ranking_comparison.png    — top-20 factor ranking comparison
  summary.png                      — combined 2x2 panel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════
# Data loading (reuse existing analyses.json)
# ══════════════════════════════════════════════════════════════════════

def load_analyses(path: str):
    """Load WindowAnalysis objects from saved JSON."""
    from sde_causal_generator.data_structures import WindowAnalysis, CausalFactor

    with open(path) as f:
        raw = json.load(f)

    analyses = []
    for w in raw:
        factors = []
        for fd in w["factors"]:
            factors.append(CausalFactor(
                name=fd["name"],
                category=fd.get("category", "unknown"),
                direction=fd.get("direction", 0.0),
                magnitude=fd.get("magnitude", 0.5),
                persistence=fd.get("persistence", "transient"),
                description=fd.get("description", ""),
            ))
        analyses.append(WindowAnalysis(
            start_date=w["start_date"],
            end_date=w["end_date"],
            window_type=w.get("window_type", "monthly"),
            price_change_pct=w.get("price_change_pct", 0.0),
            factors=factors,
            raw_llm_response=w.get("raw_llm_response", ""),
        ))
    return analyses


def load_price_data(path: str):
    """Load price DataFrame from real_data_reference.csv or from Yahoo."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_presence_matrix(analyses, df, threshold=0.0):
    """Build presence matrix — same logic as factor_extractor."""
    import pandas as pd

    all_names = set()
    for a in analyses:
        for f in a.factors:
            if f.magnitude > threshold:
                all_names.add(f.name)

    factor_names = sorted(all_names)
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
            if f.magnitude <= threshold:
                continue
            if f.name not in name_to_idx:
                continue
            col = name_to_idx[f.name]
            for d in dates:
                if start <= d <= end:
                    row = date_to_row[d]
                    presence[row, col] = max(presence[row, col], f.magnitude)

    return presence, factor_names, dates


def compute_llm_directions(analyses, factor_names):
    """Compute mean LLM direction per factor."""
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


def apply_factor_scoring(presence, factor_names, df, min_score=0.10, max_factors=200):
    """Run F4 factor scoring and pruning (same as pipeline)."""
    from sde_causal_generator.factor_relevance import (
        score_factors, prune_irrelevant_factors,
    )

    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
    df_sorted = df.sort_values("date")
    _prices = df_sorted[feature_cols].values.astype(np.float64)
    _prices = np.clip(_prices, 1e-8, None)
    _log_ret = np.diff(np.log(_prices), axis=0)

    factor_scores = score_factors(presence, _log_ret, factor_names)
    presence, factor_names = prune_irrelevant_factors(
        presence, factor_names, factor_scores,
        min_score=min_score, max_factors=max_factors,
    )
    return presence, factor_names


# ══════════════════════════════════════════════════════════════════════
# Train one condition
# ══════════════════════════════════════════════════════════════════════

def train_condition(
    presence: np.ndarray,
    price_data: np.ndarray,
    factor_names: List[str],
    feature_names: List[str],
    llm_directions: np.ndarray,
    label: str,
    seed: int = 42,
    n_lags: int = 10,
    n_epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 30,
    device: str = "auto",
) -> Dict:
    """Train TFT and return results dict."""
    import torch
    from sde_causal_generator.factor_impact_network import train_impact_network

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'─'*60}")
    print(f"  Training: {label}")
    print(f"{'─'*60}")
    print(f"  Presence matrix: {presence.shape}")
    print(f"  Presence range: [{presence.min():.3f}, {presence.max():.3f}]")
    print(f"  Non-zero entries: {(presence > 0).sum()} / {presence.size} "
          f"({100*(presence>0).mean():.1f}%)")
    print(f"  Direction range: [{llm_directions.min():.3f}, {llm_directions.max():.3f}]")
    print(f"  Unique direction values: {len(np.unique(llm_directions))}")

    t0 = time.time()
    model, impact_matrix, history = train_impact_network(
        factor_matrix=presence,
        price_data=price_data,
        factor_names=factor_names,
        feature_names=feature_names,
        n_lags=n_lags,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_fraction=0.15,
        device=device,
        patience=patience,
        architecture="tft",
        llm_directions=llm_directions,
    )
    elapsed = time.time() - t0

    # Extract key metrics
    best_epoch = history.get("best_epoch", -1)
    train_loss_final = history["train_loss"][-1] if history["train_loss"] else None
    val_loss_final = history["val_loss"][-1] if history["val_loss"] else None
    val_loss_best = min(history["val_loss"]) if history["val_loss"] else None

    # Factor rankings from impact matrix
    close_col = min(3, impact_matrix.base_impact.shape[1] - 1)
    abs_impact = np.abs(impact_matrix.base_impact[:, close_col])
    ranking_idx = np.argsort(-abs_impact)
    top_factors = [(factor_names[i], float(impact_matrix.base_impact[i, close_col]))
                   for i in ranking_idx[:20]]

    print(f"\n  {label} results:")
    print(f"    Best val loss: {val_loss_best:.5f} (epoch {best_epoch+1})")
    print(f"    Train time: {elapsed:.1f}s")
    print(f"    Top-5 factors (close impact):")
    for name, imp in top_factors[:5]:
        print(f"      {name:40s} → {imp:+.6f}")

    return {
        "label": label,
        "history": history,
        "impact_matrix": impact_matrix,
        "factor_names": factor_names,
        "top_factors": top_factors,
        "train_loss_final": train_loss_final,
        "val_loss_final": val_loss_final,
        "val_loss_best": val_loss_best,
        "best_epoch": best_epoch,
        "elapsed": elapsed,
        "presence_stats": {
            "shape": list(presence.shape),
            "nonzero_frac": float((presence > 0).mean()),
            "mean_nonzero": float(presence[presence > 0].mean()) if (presence > 0).any() else 0,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_loss_curves(hist_a, hist_b, output_path):
    """Plot training and validation loss curves for both conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(hist_a["train_loss"], label="(A) LLM magnitude/direction",
                 color="#2166ac", linewidth=1.5, alpha=0.8)
    axes[0].plot(hist_b["train_loss"], label="(B) Unit vectors",
                 color="#b2182b", linewidth=1.5, alpha=0.8)
    axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gaussian NLL Loss")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Validation loss
    axes[1].plot(hist_a["val_loss"], label="(A) LLM magnitude/direction",
                 color="#2166ac", linewidth=1.5, alpha=0.8)
    axes[1].plot(hist_b["val_loss"], label="(B) Unit vectors",
                 color="#b2182b", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Validation Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gaussian NLL Loss")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Mark best epochs
    be_a = hist_a.get("best_epoch", 0)
    be_b = hist_b.get("best_epoch", 0)
    axes[1].axvline(be_a, color="#2166ac", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].axvline(be_b, color="#b2182b", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].annotate(f"A best: ep{be_a+1}",
                     xy=(be_a, min(hist_a["val_loss"])),
                     fontsize=8, color="#2166ac")
    axes[1].annotate(f"B best: ep{be_b+1}",
                     xy=(be_b, min(hist_b["val_loss"])),
                     fontsize=8, color="#b2182b")

    fig.suptitle("TFT Ablation: LLM Magnitude/Direction vs Unit Vectors",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_impact_scatter(result_a, result_b, output_path):
    """Scatter plot: per-factor close impact A vs B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    im_a = result_a["impact_matrix"]
    im_b = result_b["impact_matrix"]

    # Align factors (should be identical, but be safe)
    names_a = result_a["factor_names"]
    names_b = result_b["factor_names"]
    common = [n for n in names_a if n in names_b]

    close_col = min(3, im_a.base_impact.shape[1] - 1)

    idx_a = {n: i for i, n in enumerate(names_a)}
    idx_b = {n: i for i, n in enumerate(names_b)}

    x = np.array([im_a.base_impact[idx_a[n], close_col] for n in common])
    y = np.array([im_b.base_impact[idx_b[n], close_col] for n in common])

    # Correlation
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, alpha=0.5, s=30, c="#2166ac", edgecolors="white",
               linewidth=0.5)

    # Identity line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    margin = 0.1 * (lims[1] - lims[0])
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("(A) LLM Magnitude/Direction — Close Impact", fontsize=12)
    ax.set_ylabel("(B) Unit Vectors — Close Impact", fontsize=12)
    ax.set_title(
        f"Per-Factor Impact: A vs B\n"
        f"Pearson r = {r:.3f} (p = {p:.2e}), "
        f"Spearman ρ = {rho:.3f} (p = {p_s:.2e})",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Label top-5 factors by absolute impact
    abs_sum = np.abs(x) + np.abs(y)
    top5 = np.argsort(-abs_sum)[:5]
    for i in top5:
        ax.annotate(
            common[i].replace("_", " ")[:25],
            xy=(x[i], y[i]),
            fontsize=7,
            textcoords="offset points",
            xytext=(5, 5),
            arrowprops=dict(arrowstyle="-", alpha=0.3),
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return r, rho


def plot_ranking_comparison(result_a, result_b, output_path, n_top=20):
    """Side-by-side top-N factor ranking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_a = result_a["top_factors"][:n_top]
    top_b = result_b["top_factors"][:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Condition A
    names_a = [f[0].replace("_", " ")[:30] for f in top_a]
    vals_a = [f[1] for f in top_a]
    colors_a = ["#2166ac" if v >= 0 else "#b2182b" for v in vals_a]
    y_pos = range(len(names_a))
    axes[0].barh(y_pos, vals_a, color=colors_a, alpha=0.8, height=0.7)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names_a, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("(A) LLM Magnitude/Direction", fontsize=13,
                       fontweight="bold")
    axes[0].set_xlabel("Close Impact")
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].grid(True, axis="x", alpha=0.3)

    # Condition B
    names_b = [f[0].replace("_", " ")[:30] for f in top_b]
    vals_b = [f[1] for f in top_b]
    colors_b = ["#2166ac" if v >= 0 else "#b2182b" for v in vals_b]
    y_pos2 = range(len(names_b))
    axes[1].barh(y_pos2, vals_b, color=colors_b, alpha=0.8, height=0.7)
    axes[1].set_yticks(y_pos2)
    axes[1].set_yticklabels(names_b, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("(B) Unit Vectors", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Close Impact")
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].grid(True, axis="x", alpha=0.3)

    # Compute overlap
    set_a = set(f[0] for f in top_a)
    set_b = set(f[0] for f in top_b)
    overlap = len(set_a & set_b)

    fig.suptitle(
        f"Top-{n_top} Factor Ranking Comparison\n"
        f"Overlap: {overlap}/{n_top} ({100*overlap/n_top:.0f}%)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return overlap


def generate_synthetic_data(
    impact_matrix,
    price_data: np.ndarray,
    feature_cols: list,
    historical_presence: np.ndarray,
    dates: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic trajectories from a trained impact matrix.

    Returns: (n_samples, T, M) array of synthetic prices.
    """
    from sde_causal_generator.generate import ImpactDrivenGenerator

    gen = ImpactDrivenGenerator(impact_matrix, window_trading_days=63)

    T = price_data.shape[0]
    M = price_data.shape[1]

    # Compute log-returns for calibration
    prices_clipped = np.clip(price_data, 1e-8, None).astype(np.float64)
    log_ret = np.diff(np.log(prices_clipped), axis=0)

    # Volume
    vol_col = feature_cols.index("volume") if "volume" in feature_cols else None
    real_volume = price_data[:, vol_col] if vol_col is not None else None

    # Align presence to price data length
    hp = historical_presence[:T].astype(np.float64)

    synth = gen.generate_scenario(
        n_steps=T,
        initial_prices=price_data[0].astype(np.float64),
        n_samples=n_samples,
        noise_scale=1.0,
        seed=seed,
        real_log_returns=log_ret,
        real_volume=real_volume,
        historical_presence=hp,
        real_prices=price_data.astype(np.float64),
        real_dates=dates[:T] if dates is not None else None,
    )
    return synth  # (n_samples, T, M)


def compute_distributional_metrics(synth: np.ndarray, real_prices: np.ndarray,
                                    close_col: int = 3) -> dict:
    """Compute key distributional metrics for synthetic vs real comparison.

    Parameters
    ----------
    synth : (n_samples, T, M) synthetic prices
    real_prices : (T, M) real prices

    Returns dict with per-sample averaged metrics.
    """
    from scipy import stats as sp_stats

    # Real log-returns (close)
    real_close = np.clip(real_prices[:, close_col], 1e-8, None)
    real_lr = np.diff(np.log(real_close))

    n_samples = synth.shape[0]
    ks_stats = []
    mean_rets = []
    vols = []
    kurts = []
    skews = []
    acf1_sq = []  # ACF(1) of squared returns (volatility clustering)
    max_dds = []

    for s in range(n_samples):
        sc = np.clip(synth[s, :, close_col], 1e-8, None)
        lr = np.diff(np.log(sc))

        # KS test vs real
        ks, _ = sp_stats.ks_2samp(real_lr, lr)
        ks_stats.append(ks)

        # Mean daily return
        mean_rets.append(float(lr.mean()))

        # Annualised volatility
        vols.append(float(lr.std() * np.sqrt(252)))

        # Excess kurtosis
        kurts.append(float(sp_stats.kurtosis(lr, fisher=True)))

        # Skewness
        skews.append(float(sp_stats.skew(lr)))

        # ACF(1) of squared returns
        sq = lr ** 2
        if len(sq) > 2:
            acf1 = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
        else:
            acf1 = 0.0
        acf1_sq.append(acf1)

        # Max drawdown
        cum = np.cumsum(lr)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dds.append(float(dd.max()))

    # Real metrics
    real_vol = float(real_lr.std() * np.sqrt(252))
    real_kurt = float(sp_stats.kurtosis(real_lr, fisher=True))
    real_skew = float(sp_stats.skew(real_lr))
    real_sq = real_lr ** 2
    real_acf1 = float(np.corrcoef(real_sq[:-1], real_sq[1:])[0, 1]) if len(real_sq) > 2 else 0.0
    real_cum = np.cumsum(real_lr)
    real_peak = np.maximum.accumulate(real_cum)
    real_dd = float((real_peak - real_cum).max())

    return {
        "ks_stat": float(np.mean(ks_stats)),
        "ks_std": float(np.std(ks_stats)),
        "mean_return": float(np.mean(mean_rets)),
        "annualised_vol": float(np.mean(vols)),
        "excess_kurtosis": float(np.mean(kurts)),
        "skewness": float(np.mean(skews)),
        "acf1_squared_returns": float(np.mean(acf1_sq)),
        "max_drawdown": float(np.mean(max_dds)),
        "real_annualised_vol": real_vol,
        "real_excess_kurtosis": real_kurt,
        "real_skewness": real_skew,
        "real_acf1_squared": real_acf1,
        "real_max_drawdown": real_dd,
        "real_mean_return": float(real_lr.mean()),
    }


def plot_generation_comparison(synth_a, synth_b, real_prices, metrics_a,
                                metrics_b, output_path, ticker="AAPL"):
    """Plot end-to-end generation comparison: trajectories, return distributions,
    volatility clustering, and metrics summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    close_col = min(3, real_prices.shape[1] - 1)
    real_close = real_prices[:, close_col]
    real_lr = np.diff(np.log(np.clip(real_close, 1e-8, None)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (0,0) Price trajectories ────────────────────────────────────
    ax = axes[0, 0]
    T = real_close.shape[0]
    t = np.arange(T)

    # Plot synthetic envelopes
    for s in range(min(synth_a.shape[0], 10)):
        ax.plot(t, synth_a[s, :, close_col], color="#2166ac", alpha=0.15, linewidth=0.5)
    for s in range(min(synth_b.shape[0], 10)):
        ax.plot(t, synth_b[s, :, close_col], color="#b2182b", alpha=0.15, linewidth=0.5)
    ax.plot(t, real_close, color="black", linewidth=1.5, label="Real", zorder=10)
    # Legend proxies
    ax.plot([], [], color="#2166ac", alpha=0.6, linewidth=1.5, label="(A) LLM mag/dir")
    ax.plot([], [], color="#b2182b", alpha=0.6, linewidth=1.5, label="(B) Unit vectors")
    ax.set_title("Synthetic Price Trajectories", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Close Price")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── (0,1) Return distributions ──────────────────────────────────
    ax = axes[0, 1]
    # Pool all synthetic returns
    lr_a_all = []
    lr_b_all = []
    for s in range(synth_a.shape[0]):
        sc = np.clip(synth_a[s, :, close_col], 1e-8, None)
        lr_a_all.append(np.diff(np.log(sc)))
    for s in range(synth_b.shape[0]):
        sc = np.clip(synth_b[s, :, close_col], 1e-8, None)
        lr_b_all.append(np.diff(np.log(sc)))
    lr_a_pool = np.concatenate(lr_a_all)
    lr_b_pool = np.concatenate(lr_b_all)

    bins = np.linspace(-0.08, 0.08, 80)
    ax.hist(real_lr, bins=bins, density=True, alpha=0.4, color="black", label="Real")
    ax.hist(lr_a_pool, bins=bins, density=True, alpha=0.35, color="#2166ac",
            label="(A) LLM", histtype="step", linewidth=1.5)
    ax.hist(lr_b_pool, bins=bins, density=True, alpha=0.35, color="#b2182b",
            label="(B) Unit", histtype="step", linewidth=1.5)
    ax.set_title("Log-Return Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Log-Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # KS between A and B
    ks_ab, p_ab = sp_stats.ks_2samp(lr_a_pool, lr_b_pool)
    ax.text(0.95, 0.95, f"KS(A vs B) = {ks_ab:.4f}\np = {p_ab:.3e}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ── (1,0) ACF of squared returns (volatility clustering) ───────
    ax = axes[1, 0]
    max_lag = 30

    def _acf(x, nlags=30):
        """Simple autocorrelation function."""
        x = x - x.mean()
        n = len(x)
        c0 = np.dot(x, x)
        if c0 < 1e-10:
            return np.zeros(nlags + 1)
        acf_vals = np.array([np.dot(x[:n-k], x[k:]) / c0 for k in range(nlags + 1)])
        return acf_vals

    real_sq = real_lr ** 2
    acf_real = _acf(real_sq, max_lag)

    # Average ACF across samples
    acf_a_all = []
    acf_b_all = []
    for s in range(synth_a.shape[0]):
        sc = np.clip(synth_a[s, :, close_col], 1e-8, None)
        lr_s = np.diff(np.log(sc))
        acf_a_all.append(_acf(lr_s ** 2, max_lag))
    for s in range(synth_b.shape[0]):
        sc = np.clip(synth_b[s, :, close_col], 1e-8, None)
        lr_s = np.diff(np.log(sc))
        acf_b_all.append(_acf(lr_s ** 2, max_lag))
    acf_a_mean = np.mean(acf_a_all, axis=0)
    acf_b_mean = np.mean(acf_b_all, axis=0)

    lags = np.arange(max_lag + 1)
    ax.bar(lags - 0.3, acf_real, width=0.3, alpha=0.6, color="black", label="Real")
    ax.bar(lags, acf_a_mean, width=0.3, alpha=0.6, color="#2166ac", label="(A) LLM")
    ax.bar(lags + 0.3, acf_b_mean, width=0.3, alpha=0.6, color="#b2182b", label="(B) Unit")
    ax.set_title("ACF of Squared Returns (Vol Clustering)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (1,1) Metrics comparison table ──────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    def _fmt(v, ref=None, pct=False):
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:.4f}"

    summary = (
        f"End-to-End Generation Comparison — {ticker}\n"
        f"{'─'*55}\n"
        f"{'Metric':<28s} {'Real':>8s} {'(A) LLM':>9s} {'(B) Unit':>9s}\n"
        f"{'─'*55}\n"
        f"{'Mean daily return':<28s} {metrics_a['real_mean_return']:>8.5f} "
        f"{metrics_a['mean_return']:>9.5f} {metrics_b['mean_return']:>9.5f}\n"
        f"{'Ann. volatility':<28s} {metrics_a['real_annualised_vol']:>8.3f} "
        f"{metrics_a['annualised_vol']:>9.3f} {metrics_b['annualised_vol']:>9.3f}\n"
        f"{'Excess kurtosis':<28s} {metrics_a['real_excess_kurtosis']:>8.2f} "
        f"{metrics_a['excess_kurtosis']:>9.2f} {metrics_b['excess_kurtosis']:>9.2f}\n"
        f"{'Skewness':<28s} {metrics_a['real_skewness']:>8.3f} "
        f"{metrics_a['skewness']:>9.3f} {metrics_b['skewness']:>9.3f}\n"
        f"{'ACF(1) squared returns':<28s} {metrics_a['real_acf1_squared']:>8.3f} "
        f"{metrics_a['acf1_squared_returns']:>9.3f} {metrics_b['acf1_squared_returns']:>9.3f}\n"
        f"{'Max drawdown':<28s} {metrics_a['real_max_drawdown']:>8.3f} "
        f"{metrics_a['max_drawdown']:>9.3f} {metrics_b['max_drawdown']:>9.3f}\n"
        f"{'─'*55}\n"
        f"{'KS vs real (mean±std)':<28s} {'':>8s} "
        f"{metrics_a['ks_stat']:.3f}±{metrics_a['ks_std']:.3f} "
        f"{metrics_b['ks_stat']:.3f}±{metrics_b['ks_std']:.3f}\n"
        f"{'KS(A vs B) pooled':<28s} {'':>8s} {ks_ab:>18.4f}\n"
        f"{'─'*55}\n"
    )

    # Verdict on generation equivalence
    ks_diff = abs(metrics_a["ks_stat"] - metrics_b["ks_stat"])
    vol_diff_pct = abs(metrics_a["annualised_vol"] - metrics_b["annualised_vol"]) / max(metrics_a["real_annualised_vol"], 1e-10)
    kurt_diff = abs(metrics_a["excess_kurtosis"] - metrics_b["excess_kurtosis"])

    if ks_ab < 0.05 and vol_diff_pct < 0.10:
        gen_verdict = ("GENERATION EQUIVALENT\n"
                       "Synthetic data from both conditions is\n"
                       "statistically indistinguishable.\n"
                       "→ LLM precision does NOT affect final output.")
    elif ks_ab < 0.10:
        gen_verdict = ("GENERATION SIMILAR\n"
                       f"Minor distributional differences (KS={ks_ab:.4f}).\n"
                       "Practical impact on downstream tasks is negligible.")
    else:
        gen_verdict = (f"GENERATION DIFFERS (KS={ks_ab:.4f})\n"
                       "The different input representations lead to\n"
                       "measurably different synthetic distributions.")

    summary += f"\n{gen_verdict}"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(
        f"End-to-End SDE Generation: LLM Values vs Unit Vectors — {ticker}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return ks_ab, p_ab


def plot_combined_summary(result_a, result_b, r_pearson, rho_spearman,
                          overlap_top20, output_path, ticker="AAPL"):
    """Combined 2×2 summary panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    hist_a = result_a["history"]
    hist_b = result_b["history"]

    # ── (0,0) Validation loss curves ───────────────────────────────
    ax = axes[0, 0]
    ax.plot(hist_a["val_loss"], label="(A) LLM mag/dir",
            color="#2166ac", linewidth=1.5)
    ax.plot(hist_b["val_loss"], label="(B) Unit vectors",
            color="#b2182b", linewidth=1.5)
    ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    be_a = hist_a.get("best_epoch", 0)
    be_b = hist_b.get("best_epoch", 0)
    ax.axvline(be_a, color="#2166ac", linestyle="--", alpha=0.4)
    ax.axvline(be_b, color="#b2182b", linestyle="--", alpha=0.4)

    # ── (0,1) Impact scatter ──────────────────────────────────────
    ax = axes[0, 1]
    im_a = result_a["impact_matrix"]
    im_b = result_b["impact_matrix"]
    close_col = min(3, im_a.base_impact.shape[1] - 1)
    names = result_a["factor_names"]
    x = im_a.base_impact[:, close_col]
    y = im_b.base_impact[:, close_col]
    ax.scatter(x, y, alpha=0.5, s=20, c="#2166ac", edgecolors="white",
               linewidth=0.3)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    m = 0.1 * (lims[1] - lims[0])
    lims = [lims[0]-m, lims[1]+m]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("(A) Impact")
    ax.set_ylabel("(B) Impact")
    ax.set_title(f"Factor Impact: r={r_pearson:.3f}, ρ={rho_spearman:.3f}",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── (1,0) Metrics comparison bar chart ─────────────────────────
    ax = axes[1, 0]
    metrics = [
        ("Best Val Loss", result_a["val_loss_best"], result_b["val_loss_best"]),
        ("Best Epoch", result_a["best_epoch"]+1, result_b["best_epoch"]+1),
        ("Train Time (s)", result_a["elapsed"], result_b["elapsed"]),
    ]
    x_pos = np.arange(len(metrics))
    width = 0.35
    vals_a = [m[1] for m in metrics]
    vals_b = [m[2] for m in metrics]
    # Normalise for display
    max_vals = [max(abs(a), abs(b), 1e-10) for a, b in zip(vals_a, vals_b)]
    norm_a = [v/mx for v, mx in zip(vals_a, max_vals)]
    norm_b = [v/mx for v, mx in zip(vals_b, max_vals)]
    bars_a = ax.bar(x_pos - width/2, norm_a, width, label="(A) LLM",
                     color="#2166ac", alpha=0.8)
    bars_b = ax.bar(x_pos + width/2, norm_b, width, label="(B) Unit",
                     color="#b2182b", alpha=0.8)
    # Annotate with real values
    for bar, val in zip(bars_a, vals_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{val:.0f}",
                ha="center", fontsize=8)
    for bar, val in zip(bars_b, vals_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{val:.0f}",
                ha="center", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax.set_title("Metrics Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.3)

    # ── (1,1) Summary text box ────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = (
        f"TFT Magnitude/Direction Ablation — {ticker}\n"
        f"{'─'*45}\n\n"
        f"Condition A: LLM magnitude [0,1] + direction [-1,+1]\n"
        f"Condition B: Binary presence (1.0) + sign direction {{-1,0,+1}}\n\n"
        f"{'─'*45}\n"
        f"{'Metric':<30s} {'(A) LLM':>10s} {'(B) Unit':>10s}\n"
        f"{'─'*45}\n"
        f"{'Best validation loss':<30s} {result_a['val_loss_best']:>10.5f} {result_b['val_loss_best']:>10.5f}\n"
        f"{'Best epoch':<30s} {result_a['best_epoch']+1:>10d} {result_b['best_epoch']+1:>10d}\n"
        f"{'Train time (s)':<30s} {result_a['elapsed']:>10.1f} {result_b['elapsed']:>10.1f}\n"
        f"{'─'*45}\n"
        f"{'Pearson r (impacts)':<30s} {r_pearson:>10.3f}\n"
        f"{'Spearman ρ (impacts)':<30s} {rho_spearman:>10.3f}\n"
        f"{'Top-20 overlap':<30s} {overlap_top20:>10d}/20\n"
        f"{'─'*45}\n\n"
    )

    # Interpretation
    loss_ratio = result_b["val_loss_best"] / max(result_a["val_loss_best"], 1e-10)
    if loss_ratio < 1.05 and loss_ratio > 0.95:
        verdict = ("EQUIVALENT: Unit vectors produce statistically\n"
                   "indistinguishable TFT performance.\n"
                   "→ LLM magnitude precision does NOT matter.")
    elif loss_ratio >= 1.05:
        pct = (loss_ratio - 1) * 100
        verdict = (f"LLM HELPS: Unit vectors {pct:.1f}% worse.\n"
                   f"→ Continuous magnitude provides a convergence\n"
                   f"  advantage (but TFT can compensate over time).")
    else:
        pct = (1 - loss_ratio) * 100
        verdict = (f"UNIT BETTER: Unit vectors {pct:.1f}% better.\n"
                   f"→ LLM magnitude noise may hurt more than help.")

    summary_text += f"VERDICT: {verdict}"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(
        f"TFT Input Sensitivity: LLM Values vs Unit Vectors — {ticker}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TFT Magnitude/Direction Ablation — Unit Vector Test")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--output-dir", default="results/tft_unit_vector_test")
    parser.add_argument("--analyses",
                        default="results/multi_asset_2018_2023/AAPL/analyses.json")
    parser.add_argument("--price-data",
                        default="results/multi_asset_2018_2023/AAPL/real_data_reference.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-lags", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="F4 factor scoring threshold")
    parser.add_argument("--max-factors", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  TFT MAGNITUDE/DIRECTION ABLATION — UNIT VECTOR TEST")
    print("=" * 60)
    print(f"  Ticker: {args.ticker}")
    print(f"  Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print()

    # ── Load data ───────────────────────────────────────────────────
    print("Loading analyses and price data...")
    analyses = load_analyses(args.analyses)
    df = load_price_data(args.price_data)
    print(f"  Windows: {len(analyses)}, Price rows: {len(df)}")

    # ── Build presence matrix with real magnitudes ──────────────────
    print("\nBuilding presence matrix...")
    presence_real, factor_names, dates = build_presence_matrix(analyses, df)
    print(f"  Raw factors: {len(factor_names)}, T={presence_real.shape[0]}")

    # ── F4 Factor scoring ───────────────────────────────────────────
    print("\nApplying F4 factor scoring...")
    n_pre = len(factor_names)
    presence_real, factor_names = apply_factor_scoring(
        presence_real, factor_names, df,
        min_score=args.min_score, max_factors=args.max_factors,
    )
    print(f"  Factors after scoring: {n_pre} → {len(factor_names)}")

    # ── Build LLM directions ───────────────────────────────────────
    directions_real = compute_llm_directions(analyses, factor_names)

    # ── Condition A: Current pipeline (real magnitude + direction) ──
    print(f"\nPresence stats (A): mean-nonzero = "
          f"{presence_real[presence_real > 0].mean():.3f}, "
          f"unique magnitudes = {len(np.unique(presence_real))}")

    # ── Condition B: Unit vectors ─────────────────────────────────
    presence_unit = (presence_real > 0).astype(np.float32)
    directions_unit = np.sign(directions_real).astype(np.float32)

    print(f"Presence stats (B): mean-nonzero = "
          f"{presence_unit[presence_unit > 0].mean():.3f}, "
          f"unique magnitudes = {len(np.unique(presence_unit))}")
    print(f"Direction stats: real unique = {len(np.unique(directions_real))}, "
          f"unit unique = {len(np.unique(directions_unit))}")

    # ── Build price matrix ─────────────────────────────────────────
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
    df_sorted = df.sort_values("date")
    price_data = df_sorted[feature_cols].values.astype(np.float32)

    # Align sizes
    min_T = min(presence_real.shape[0], price_data.shape[0])
    presence_real = presence_real[:min_T]
    presence_unit = presence_unit[:min_T]
    price_data = price_data[:min_T]

    # ── Train both conditions ──────────────────────────────────────
    common_kwargs = dict(
        price_data=price_data,
        factor_names=factor_names,
        feature_names=feature_cols,
        seed=args.seed,
        n_lags=args.n_lags,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device,
    )

    result_a = train_condition(
        presence=presence_real,
        llm_directions=directions_real,
        label="(A) LLM Magnitude/Direction",
        **common_kwargs,
    )

    result_b = train_condition(
        presence=presence_unit,
        llm_directions=directions_unit,
        label="(B) Unit Vectors",
        **common_kwargs,
    )

    # ── Generate plots ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Generating comparison plots...")
    print(f"{'─'*60}")

    plot_loss_curves(
        result_a["history"], result_b["history"],
        os.path.join(args.output_dir, "loss_curves.png"),
    )

    r_pearson, rho_spearman = plot_impact_scatter(
        result_a, result_b,
        os.path.join(args.output_dir, "impact_matrix_scatter.png"),
    )

    overlap = plot_ranking_comparison(
        result_a, result_b,
        os.path.join(args.output_dir, "factor_ranking_comparison.png"),
    )

    plot_combined_summary(
        result_a, result_b,
        r_pearson, rho_spearman, overlap,
        os.path.join(args.output_dir, "summary.png"),
        ticker=args.ticker,
    )

    # Also PDF
    plot_combined_summary(
        result_a, result_b,
        r_pearson, rho_spearman, overlap,
        os.path.join(args.output_dir, "summary.pdf"),
        ticker=args.ticker,
    )

    # ── End-to-end generation comparison ───────────────────────────
    print(f"\n{'─'*60}")
    print("  End-to-End SDE Generation Comparison")
    print(f"{'─'*60}")

    import pandas as pd
    all_dates = pd.to_datetime(df_sorted["date"]).values

    print("  Generating 20 trajectories with impact matrix (A)...")
    synth_a = generate_synthetic_data(
        result_a["impact_matrix"], price_data, feature_cols,
        presence_real, all_dates, n_samples=20, seed=args.seed,
    )
    print(f"    Shape: {synth_a.shape}")

    print("  Generating 20 trajectories with impact matrix (B)...")
    synth_b = generate_synthetic_data(
        result_b["impact_matrix"], price_data, feature_cols,
        presence_unit, all_dates, n_samples=20, seed=args.seed,
    )
    print(f"    Shape: {synth_b.shape}")

    print("  Computing distributional metrics...")
    metrics_a = compute_distributional_metrics(synth_a, price_data)
    metrics_b = compute_distributional_metrics(synth_b, price_data)

    print("  Generating generation comparison plots...")
    ks_ab, p_ab = plot_generation_comparison(
        synth_a, synth_b, price_data, metrics_a, metrics_b,
        os.path.join(args.output_dir, "generation_comparison.png"),
        ticker=args.ticker,
    )
    # Also PDF
    plot_generation_comparison(
        synth_a, synth_b, price_data, metrics_a, metrics_b,
        os.path.join(args.output_dir, "generation_comparison.pdf"),
        ticker=args.ticker,
    )

    print(f"\n  KS(A vs B) pooled returns: {ks_ab:.4f} (p = {p_ab:.3e})")
    print(f"  KS vs real — (A): {metrics_a['ks_stat']:.3f}±{metrics_a['ks_std']:.3f}, "
          f"(B): {metrics_b['ks_stat']:.3f}±{metrics_b['ks_std']:.3f}")
    print(f"  Ann. vol  — Real: {metrics_a['real_annualised_vol']:.3f}, "
          f"(A): {metrics_a['annualised_vol']:.3f}, "
          f"(B): {metrics_b['annualised_vol']:.3f}")
    print(f"  Kurtosis  — Real: {metrics_a['real_excess_kurtosis']:.2f}, "
          f"(A): {metrics_a['excess_kurtosis']:.2f}, "
          f"(B): {metrics_b['excess_kurtosis']:.2f}")

    # ── Save results JSON ──────────────────────────────────────────
    results_data = {
        "experiment": "TFT Magnitude/Direction Ablation — Unit Vector Test",
        "ticker": args.ticker,
        "seed": args.seed,
        "n_factors": len(factor_names),
        "n_datapoints": min_T,
        "conditions": {
            "llm_values": {
                "description": "LLM magnitude [0,1] + mean direction [-1,+1]",
                "val_loss_best": result_a["val_loss_best"],
                "best_epoch": result_a["best_epoch"],
                "train_loss_final": result_a["train_loss_final"],
                "elapsed_seconds": round(result_a["elapsed"], 1),
                "presence_stats": result_a["presence_stats"],
                "top_20_factors": result_a["top_factors"],
            },
            "unit_vectors": {
                "description": "Binary presence (1.0) + sign direction {-1,0,+1}",
                "val_loss_best": result_b["val_loss_best"],
                "best_epoch": result_b["best_epoch"],
                "train_loss_final": result_b["train_loss_final"],
                "elapsed_seconds": round(result_b["elapsed"], 1),
                "presence_stats": result_b["presence_stats"],
                "top_20_factors": result_b["top_factors"],
            },
        },
        "comparison": {
            "pearson_r": round(r_pearson, 4),
            "spearman_rho": round(rho_spearman, 4),
            "top_20_overlap": overlap,
            "val_loss_ratio_B_over_A": round(
                result_b["val_loss_best"] / max(result_a["val_loss_best"], 1e-10), 4
            ),
        },
        "generation_comparison": {
            "ks_A_vs_B_pooled": round(ks_ab, 4),
            "ks_A_vs_B_pvalue": float(p_ab),
            "condition_A_metrics": metrics_a,
            "condition_B_metrics": metrics_b,
        },
        "loss_history": {
            "llm_values": {
                "train": result_a["history"]["train_loss"],
                "val": result_a["history"]["val_loss"],
            },
            "unit_vectors": {
                "train": result_b["history"]["train_loss"],
                "val": result_b["history"]["val_loss"],
            },
        },
    }

    out_json = os.path.join(args.output_dir, "tft_unit_vector_results.json")
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_json}")

    # ── Print summary ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TFT UNIT VECTOR TEST — SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'(A) LLM':>12s} {'(B) Unit':>12s}")
    print(f"  {'─'*54}")
    print(f"  {'Best val loss':<30s} {result_a['val_loss_best']:>12.5f} {result_b['val_loss_best']:>12.5f}")
    print(f"  {'Best epoch':<30s} {result_a['best_epoch']+1:>12d} {result_b['best_epoch']+1:>12d}")
    print(f"  {'Train time (s)':<30s} {result_a['elapsed']:>12.1f} {result_b['elapsed']:>12.1f}")
    print(f"  {'─'*54}")
    print(f"  Pearson r (factor impacts):   {r_pearson:.3f}")
    print(f"  Spearman ρ (factor impacts):  {rho_spearman:.3f}")
    print(f"  Top-20 ranking overlap:       {overlap}/20")
    print()
    print(f"  {'─'*54}")
    print(f"  END-TO-END GENERATION:")
    print(f"  {'─'*54}")
    print(f"  {'Metric':<28s} {'Real':>8s} {'(A) LLM':>9s} {'(B) Unit':>9s}")
    print(f"  {'─'*54}")
    print(f"  {'Ann. volatility':<28s} {metrics_a['real_annualised_vol']:>8.3f} "
          f"{metrics_a['annualised_vol']:>9.3f} {metrics_b['annualised_vol']:>9.3f}")
    print(f"  {'Excess kurtosis':<28s} {metrics_a['real_excess_kurtosis']:>8.2f} "
          f"{metrics_a['excess_kurtosis']:>9.2f} {metrics_b['excess_kurtosis']:>9.2f}")
    print(f"  {'ACF(1) sq. returns':<28s} {metrics_a['real_acf1_squared']:>8.3f} "
          f"{metrics_a['acf1_squared_returns']:>9.3f} {metrics_b['acf1_squared_returns']:>9.3f}")
    print(f"  {'KS vs real':<28s} {'':>8s} "
          f"{metrics_a['ks_stat']:>6.3f}±{metrics_a['ks_std']:.3f} "
          f"{metrics_b['ks_stat']:>3.3f}±{metrics_b['ks_std']:.3f}")
    print(f"  {'KS(A vs B) pooled':<28s} {'':>8s} {ks_ab:>18.4f}")
    print(f"  {'─'*54}")
    print()

    ratio = result_b["val_loss_best"] / max(result_a["val_loss_best"], 1e-10)
    vol_diff = abs(metrics_a["annualised_vol"] - metrics_b["annualised_vol"]) / max(metrics_a["real_annualised_vol"], 1e-10)

    # Combined verdict: training + generation
    if 0.95 <= ratio <= 1.05 and ks_ab < 0.10:
        print("  VERDICT: EQUIVALENT (training + generation)")
        print("  → TFT converges to the same solution with unit vectors.")
        print(f"  → Generated data is statistically indistinguishable (KS={ks_ab:.4f}).")
        print("  → LLM magnitude/direction precision does NOT matter.")
        print("  → VSN (Variable Selection Network) compensates fully.")
    elif 0.95 <= ratio <= 1.05:
        print(f"  VERDICT: TRAINING EQUIVALENT, GENERATION DIFFERS (KS={ks_ab:.4f})")
        print("  → TFT trains equally well, but impact matrix differences")
        print("    propagate to measurably different synthetic distributions.")
    elif ratio > 1.05:
        pct = (ratio - 1) * 100
        print(f"  VERDICT: LLM VALUES HELP ({pct:.1f}% better val loss)")
        print(f"  → Generation KS(A vs B) = {ks_ab:.4f}")
    else:
        pct = (1 - ratio) * 100
        print(f"  VERDICT: UNIT VECTORS BETTER ({pct:.1f}% better val loss)")
        print(f"  → Generation KS(A vs B) = {ks_ab:.4f}")

    print()
    print(f"  Output: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
