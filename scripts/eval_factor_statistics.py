#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Term Statistics — LLM Extraction Analysis.

Analyses the extracted causal factors from the pipeline run and produces:

1. Term frequency analysis:
   - Most common factor names across all tickers and windows
   - Factor name word cloud data (term → frequency)
   - Category distribution

2. Per-ticker statistics:
   - Number of unique factors
   - Number of windows
   - Factors per window (mean, std)
   - Direction distribution (bullish vs bearish)

3. Cross-ticker factor overlap:
   - Jaccard similarity of factor sets between tickers
   - Common factors that appear in multiple tickers

4. Temporal patterns:
   - Factor density over time
   - Bull/bear factor ratio over time

Output:
  results/final_run_gpt4o_mini/factor_statistics/
    factor_statistics.json
    factor_term_frequency.csv
    factor_category_distribution.png
    factor_term_wordcloud.png
    factor_cross_ticker_heatmap.png
    factor_temporal_density.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_analyses(results_dir: str, tickers: List[str]) -> Dict[str, List[Dict]]:
    """Load analyses.json for each ticker."""
    all_analyses = {}
    for ticker in tickers:
        path = os.path.join(results_dir, ticker, "analyses.json")
        if os.path.exists(path):
            with open(path) as f:
                all_analyses[ticker] = json.load(f)
            print(f"  Loaded {ticker}: {len(all_analyses[ticker])} windows")
        else:
            print(f"  WARNING: {path} not found")
    return all_analyses


def extract_all_factors(all_analyses: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Flatten all factors into a single DataFrame."""
    rows = []
    for ticker, analyses in all_analyses.items():
        for window in analyses:
            for f in window.get("factors", []):
                rows.append({
                    "ticker": ticker,
                    "start_date": window.get("start_date", ""),
                    "end_date": window.get("end_date", ""),
                    "price_change_pct": window.get("price_change_pct", 0),
                    "factor_name": f.get("name", ""),
                    "category": f.get("category", "unknown"),
                    "direction": f.get("direction", 0),
                    "magnitude": f.get("magnitude", 0),
                    "confidence": f.get("confidence", 0),
                    "description": f.get("description", ""),
                })
    return pd.DataFrame(rows)


def tokenize_factor_name(name: str) -> List[str]:
    """Split factor name into meaningful tokens."""
    stop_words = {
        "the", "a", "an", "of", "in", "on", "to", "for", "and", "is",
        "market", "stock", "price", "impact", "effect", "factor",
        "general", "overall", "related",
    }
    tokens = re.split(r'[_\-\s]+', name.lower())
    return [t for t in tokens if t and t not in stop_words and len(t) > 1]


def compute_term_frequency(factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute word-level term frequency across all factor names."""
    all_tokens = []
    for name in factors_df["factor_name"]:
        all_tokens.extend(tokenize_factor_name(name))
    counter = Counter(all_tokens)
    df = pd.DataFrame(counter.most_common(), columns=["term", "frequency"])
    df["relative_frequency"] = df["frequency"] / df["frequency"].sum()
    return df


def compute_factor_frequency(factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute factor-name-level frequency."""
    # Exclude noise factors
    filtered = factors_df[
        ~factors_df["factor_name"].str.contains("noise|consolidation", case=False)
    ]
    counter = Counter(filtered["factor_name"])
    df = pd.DataFrame(counter.most_common(), columns=["factor_name", "occurrences"])
    # Add mean direction and category
    stats = []
    for _, row in df.iterrows():
        mask = filtered["factor_name"] == row["factor_name"]
        subset = filtered[mask]
        stats.append({
            "mean_direction": round(float(subset["direction"].mean()), 3),
            "mean_magnitude": round(float(subset["magnitude"].mean()), 3),
            "n_tickers": subset["ticker"].nunique(),
            "tickers": sorted(subset["ticker"].unique().tolist()),
            "most_common_category": subset["category"].mode().iloc[0] if len(subset) > 0 else "unknown",
        })
    stats_df = pd.DataFrame(stats)
    return pd.concat([df, stats_df], axis=1)


def compute_per_ticker_stats(
    all_analyses: Dict[str, List[Dict]],
    factors_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """Per-ticker summary statistics."""
    stats = {}
    for ticker, analyses in all_analyses.items():
        ticker_factors = factors_df[factors_df["ticker"] == ticker]
        noise_mask = ticker_factors["factor_name"].str.contains(
            "noise|consolidation", case=False
        )
        non_noise = ticker_factors[~noise_mask]

        factors_per_window = []
        for w in analyses:
            n_real = len([
                f for f in w.get("factors", [])
                if "noise" not in f.get("name", "") and "consolidation" not in f.get("name", "")
            ])
            factors_per_window.append(n_real)

        stats[ticker] = {
            "n_windows": len(analyses),
            "total_factors": len(ticker_factors),
            "unique_factors": non_noise["factor_name"].nunique(),
            "factors_per_window_mean": round(float(np.mean(factors_per_window)), 1),
            "factors_per_window_std": round(float(np.std(factors_per_window)), 1),
            "n_bullish": int((non_noise["direction"] > 0.15).sum()),
            "n_bearish": int((non_noise["direction"] < -0.15).sum()),
            "n_neutral": int(
                ((non_noise["direction"] >= -0.15) & (non_noise["direction"] <= 0.15)).sum()
            ),
            "mean_magnitude": round(float(non_noise["magnitude"].mean()), 3),
            "mean_confidence": round(float(non_noise["confidence"].mean()), 3),
            "category_distribution": dict(non_noise["category"].value_counts()),
        }
    return stats


def compute_cross_ticker_overlap(
    factors_df: pd.DataFrame,
    tickers: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Jaccard similarity of factor name sets between tickers."""
    noise_mask = factors_df["factor_name"].str.contains(
        "noise|consolidation", case=False
    )
    filtered = factors_df[~noise_mask]

    factor_sets = {}
    for ticker in tickers:
        factor_sets[ticker] = set(
            filtered[filtered["ticker"] == ticker]["factor_name"].unique()
        )

    n = len(tickers)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si = factor_sets[tickers[i]]
            sj = factor_sets[tickers[j]]
            if si or sj:
                jaccard[i, j] = len(si & sj) / len(si | sj)
            else:
                jaccard[i, j] = 0.0
    return jaccard, tickers


# ════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════

def plot_category_distribution(factors_df: pd.DataFrame, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_mask = factors_df["factor_name"].str.contains(
        "noise|consolidation", case=False
    )
    filtered = factors_df[~noise_mask]
    cat_counts = filtered["category"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_counts)))
    bars = ax.barh(cat_counts.index, cat_counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Factor Occurrences", fontsize=12)
    ax.set_title("Factor Category Distribution (All Tickers)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    for bar, v in zip(bars, cat_counts.values):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_term_wordcloud(term_freq: pd.DataFrame, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from wordcloud import WordCloud
        word_dict = dict(zip(term_freq["term"], term_freq["frequency"]))
        wc = WordCloud(
            width=1200, height=600, background_color="white",
            colormap="viridis", max_words=100,
        ).generate_from_frequencies(word_dict)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Factor Term Word Cloud", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")
    except ImportError:
        # Fallback: horizontal bar chart of top 30 terms
        top = term_freq.head(30)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["term"][::-1], top["frequency"][::-1],
                color="#3498db", edgecolor="white")
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_title("Top 30 Factor Terms", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path} (bar chart fallback)")


def plot_cross_ticker_heatmap(
    jaccard: np.ndarray,
    tickers: List[str],
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n = len(tickers)
    colors = ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    cmap = LinearSegmentedColormap.from_list("reds", colors, N=256)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title("Cross-Ticker Factor Overlap\n(Jaccard Similarity)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(tickers, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = jaccard[i, j]
            color = "white" if val > 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Index")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_temporal_density(
    all_analyses: Dict[str, List[Dict]],
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Factor count over time (stacked by ticker)
    ax = axes[0]
    for ticker, analyses in sorted(all_analyses.items()):
        dates = []
        counts = []
        for w in analyses:
            try:
                mid_date = pd.Timestamp(w["start_date"]) + (
                    pd.Timestamp(w["end_date"]) - pd.Timestamp(w["start_date"])
                ) / 2
                n_real = len([
                    f for f in w.get("factors", [])
                    if "noise" not in f.get("name", "") and "consolidation" not in f.get("name", "")
                ])
                dates.append(mid_date)
                counts.append(n_real)
            except Exception:
                pass
        if dates:
            ax.plot(dates, counts, label=ticker, alpha=0.7, linewidth=1.5)

    ax.set_ylabel("Factors per Window", fontsize=12)
    ax.set_title("Factor Extraction Density Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Bull/Bear ratio over time (aggregated)
    ax = axes[1]
    all_dates = []
    all_ratios = []
    for ticker, analyses in all_analyses.items():
        for w in analyses:
            try:
                mid_date = pd.Timestamp(w["start_date"]) + (
                    pd.Timestamp(w["end_date"]) - pd.Timestamp(w["start_date"])
                ) / 2
                factors = [
                    f for f in w.get("factors", [])
                    if "noise" not in f.get("name", "") and "consolidation" not in f.get("name", "")
                ]
                if factors:
                    n_bull = sum(1 for f in factors if f.get("direction", 0) > 0.15)
                    n_bear = sum(1 for f in factors if f.get("direction", 0) < -0.15)
                    ratio = n_bull / max(n_bull + n_bear, 1)
                    all_dates.append(mid_date)
                    all_ratios.append(ratio)
            except Exception:
                pass

    if all_dates:
        # Sort by date and compute rolling average
        sorted_idx = np.argsort(all_dates)
        sorted_dates = [all_dates[i] for i in sorted_idx]
        sorted_ratios = [all_ratios[i] for i in sorted_idx]

        ax.scatter(sorted_dates, sorted_ratios, alpha=0.15, s=10, c="#666")
        # Rolling mean
        window = min(50, len(sorted_ratios) // 3)
        if window > 1:
            rolling = pd.Series(sorted_ratios).rolling(window, center=True).mean()
            ax.plot(sorted_dates, rolling, color="#e74c3c", linewidth=2,
                    label=f"Rolling mean (w={window})")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Balanced")
    ax.set_ylabel("Bullish Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Bullish vs Bearish Factor Ratio Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, 1.05)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_direction_magnitude_scatter(factors_df: pd.DataFrame, output_path: str):
    """Scatter plot of factor direction vs magnitude, colored by category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_mask = factors_df["factor_name"].str.contains(
        "noise|consolidation", case=False
    )
    filtered = factors_df[~noise_mask]

    categories = sorted(filtered["category"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    cat_color = dict(zip(categories, colors))

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in categories:
        subset = filtered[filtered["category"] == cat]
        ax.scatter(
            subset["direction"], subset["magnitude"],
            c=[cat_color[cat]], label=cat, alpha=0.4, s=20, edgecolors="white",
            linewidths=0.3,
        )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Direction (−1 bearish → +1 bullish)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("Factor Direction vs Magnitude by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Factor Term Statistics — LLM Extraction Analysis")
    parser.add_argument("--results-dir",
                        default="results/final_run_gpt4o_mini")
    parser.add_argument("--output-dir",
                        default="results/final_run_gpt4o_mini/factor_statistics")
    parser.add_argument("--tickers", nargs="+",
                        default=["AAPL", "MSFT", "JPM", "XOM",
                                 "JNJ", "WMT", "CAT", "AMGN"])
    args = parser.parse_args()

    results_dir = os.path.join(str(PROJECT_ROOT), args.results_dir)
    output_dir = os.path.join(str(PROJECT_ROOT), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("FACTOR TERM STATISTICS — LLM EXTRACTION ANALYSIS")
    print("=" * 70)
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Tickers:     {args.tickers}")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────
    print("\n1. Loading analyses...")
    all_analyses = load_all_analyses(results_dir, args.tickers)
    if not all_analyses:
        print("ERROR: No analyses found.")
        sys.exit(1)

    # ── Extract all factors ─────────────────────────────────────────
    print("\n2. Extracting factors...")
    factors_df = extract_all_factors(all_analyses)
    print(f"  Total factor occurrences: {len(factors_df)}")
    print(f"  Unique factor names:      {factors_df['factor_name'].nunique()}")

    # ── Term frequency ──────────────────────────────────────────────
    print("\n3. Computing term frequency...")
    term_freq = compute_term_frequency(factors_df)
    term_freq.to_csv(os.path.join(output_dir, "factor_term_frequency.csv"),
                     index=False)
    print(f"  Top 20 terms:")
    for _, row in term_freq.head(20).iterrows():
        print(f"    {row['term']:>20s}  {row['frequency']:>5d}  "
              f"({row['relative_frequency']:.1%})")

    # ── Factor frequency ────────────────────────────────────────────
    print("\n4. Computing factor frequency...")
    factor_freq = compute_factor_frequency(factors_df)
    factor_freq.to_csv(os.path.join(output_dir, "factor_name_frequency.csv"),
                       index=False)
    print(f"  Top 20 factors:")
    for _, row in factor_freq.head(20).iterrows():
        dir_str = "↑" if row["mean_direction"] > 0.15 else (
            "↓" if row["mean_direction"] < -0.15 else "→"
        )
        print(f"    {row['factor_name']:>40s}  {row['occurrences']:>4d}  "
              f"{dir_str} d={row['mean_direction']:+.2f}  "
              f"tickers={row['n_tickers']}")

    # ── Per-ticker stats ────────────────────────────────────────────
    print("\n5. Computing per-ticker statistics...")
    per_ticker = compute_per_ticker_stats(all_analyses, factors_df)
    print(f"  {'Ticker':>6s}  {'Windows':>7s}  {'Unique':>6s}  "
          f"{'F/Win':>5s}  {'Bull':>5s}  {'Bear':>5s}")
    for ticker in args.tickers:
        if ticker in per_ticker:
            s = per_ticker[ticker]
            print(f"  {ticker:>6s}  {s['n_windows']:>7d}  {s['unique_factors']:>6d}  "
                  f"{s['factors_per_window_mean']:>5.1f}  "
                  f"{s['n_bullish']:>5d}  {s['n_bearish']:>5d}")

    # ── Cross-ticker overlap ────────────────────────────────────────
    print("\n6. Computing cross-ticker overlap...")
    active_tickers = [t for t in args.tickers if t in all_analyses]
    jaccard, ticker_order = compute_cross_ticker_overlap(factors_df, active_tickers)

    # ── Save comprehensive JSON ─────────────────────────────────────
    print("\n7. Saving results...")
    results = {
        "summary": {
            "total_factor_occurrences": len(factors_df),
            "unique_factor_names": int(factors_df["factor_name"].nunique()),
            "unique_terms": len(term_freq),
            "n_tickers": len(all_analyses),
            "n_windows_total": sum(len(a) for a in all_analyses.values()),
        },
        "per_ticker": per_ticker,
        "top_30_terms": term_freq.head(30).to_dict(orient="records"),
        "top_50_factors": factor_freq.head(50).to_dict(orient="records"),
        "category_distribution": dict(
            factors_df[
                ~factors_df["factor_name"].str.contains("noise|consolidation", case=False)
            ]["category"].value_counts()
        ),
        "cross_ticker_jaccard": {
            "tickers": ticker_order,
            "matrix": jaccard.tolist(),
        },
    }

    out_json = os.path.join(output_dir, "factor_statistics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_json}")

    # ── Generate plots ──────────────────────────────────────────────
    print("\n8. Generating plots...")
    plot_category_distribution(
        factors_df,
        os.path.join(output_dir, "factor_category_distribution.png"),
    )
    plot_term_wordcloud(
        term_freq,
        os.path.join(output_dir, "factor_term_wordcloud.png"),
    )
    plot_cross_ticker_heatmap(
        jaccard, ticker_order,
        os.path.join(output_dir, "factor_cross_ticker_heatmap.png"),
    )
    plot_temporal_density(
        all_analyses,
        os.path.join(output_dir, "factor_temporal_density.png"),
    )
    plot_direction_magnitude_scatter(
        factors_df,
        os.path.join(output_dir, "factor_direction_magnitude.png"),
    )

    # ── Save raw factors as CSV for reproducibility ─────────────────
    factors_df.to_csv(
        os.path.join(output_dir, "all_factors_raw.csv"), index=False
    )
    print(f"  Saved: all_factors_raw.csv")

    print(f"\n{'=' * 70}")
    print(f"DONE — Factor statistics saved to {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
