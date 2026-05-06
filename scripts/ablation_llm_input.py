#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Ablation Test — Factor Extraction Input Sensitivity.

Tests whether LLMs extract factors based on NEWS (RAG), PRICE DATA, or both.

Three conditions (same 20 windows, same 5 LLMs):
  (A) Price + RAG   — full pipeline (reused from cross-LLM semantic test)
  (B) RAG only      — news context provided, price data REMOVED from prompt
  (C) Price only    — price data provided, news context REMOVED from prompt

For each condition:
  - Pairwise directional concordance heatmap (like the reference image)
  - Fleiss' κ across all 5 LLMs

Output:
  results/ablation_llm_test/
    ablation_results.json
    heatmap_A_price_and_rag.png
    heatmap_B_rag_only.png
    heatmap_C_price_only.png
    heatmap_combined.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

FACTOR_CATEGORIES = [
    "macroeconomic", "geopolitical", "sector_specific",
    "company_specific", "market_sentiment", "technical", "liquidity",
]

# ── Models (same as cross-LLM semantic test) ────────────────────────
MODELS = [
    ("Gemma-3-27B",       "openrouter/google/gemma-3-27b-it"),
    ("GPT-4o-mini",       "openrouter/openai/gpt-4o-mini"),
    ("Claude-3.5-Haiku",  "openrouter/anthropic/claude-3.5-haiku"),
    ("Mistral-Small-3.1", "openrouter/mistralai/mistral-small-3.1-24b-instruct"),
    ("Gemini-2.0-Flash",  "openrouter/google/gemini-2.0-flash-001"),
]

MODEL_SHORT = {
    "Gemma-3-27B": "Gemma-3",
    "GPT-4o-mini": "GPT-4o-mini",
    "Claude-3.5-Haiku": "Claude-3",
    "Mistral-Small-3.1": "Mistral-3.2",
    "Gemini-2.0-Flash": "Gemini-3.1",
}


# ══════════════════════════════════════════════════════════════════════
# Prompts for each condition
# ══════════════════════════════════════════════════════════════════════

_PROMPT_PRICE_AND_RAG = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Identify the specific factors that CAUSED
the price movement — not factors that merely coincided.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

PRICE DATA:
- Opening price: {open_price}
- Closing price: {close_price}
- Price change: {price_change_pct:.2f}%

{news_context}

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. If no relevant news explains the movement, attribute to "market_noise".
3. Do NOT reference events NOT in the provided news context.
4. Do NOT use knowledge after {end_date}.
5. direction means the FACTOR's intrinsic effect on {ticker} — NOT the
   price change itself.  A bearish event is bearish (direction < 0)
   even if the stock went UP because of other offsetting factors.

Return a JSON array.  Each factor:
- "name": short snake_case (e.g. "fed_rate_hike")
- "category": one of {categories}
- "direction": float -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float 0.0 to 1.0
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation citing evidence

Magnitudes MUST sum to 1.0.  Include "market_noise" for unexplained variance.
Respond ONLY with valid JSON — no markdown, no commentary.
"""

_PROMPT_RAG_ONLY = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Identify the specific factors that likely
affected {ticker} stock during the period below — based ONLY on the
news articles provided.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

NOTE: No price data is provided.  You must infer the likely market
impact of each factor purely from the news content, your financial
knowledge, and the economic context of the period.

{news_context}

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. If no articles seem relevant, return only a "market_noise" factor.
3. Do NOT reference events NOT in the provided news context.
4. Do NOT use knowledge of what happened AFTER {end_date}.
5. direction means the FACTOR's intrinsic effect on {ticker}.
   A bearish event is bearish (direction < 0) regardless of what
   the stock price actually did.

Return a JSON array.  Each factor:
- "name": short snake_case (e.g. "fed_rate_hike")
- "category": one of {categories}
- "direction": float -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float 0.0 to 1.0
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation citing evidence

Magnitudes MUST sum to 1.0.  Include "market_noise" for unexplained variance.
Respond ONLY with valid JSON — no markdown, no commentary.
"""

_PROMPT_PRICE_ONLY = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Identify the specific factors that likely
CAUSED the price movement shown below.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

PRICE DATA:
- Opening price: {open_price}
- Closing price: {close_price}
- Price change: {price_change_pct:.2f}%

NOTE: No news articles are available for this period.  You must infer
the likely causes based on your knowledge of what was publicly known
DURING this period (not after).

CRITICAL RULES:
1. Only cite factors that were KNOWN and OBSERVABLE during this period.
2. Do NOT use hindsight from future events.
3. If the movement was likely random noise, say so explicitly.
4. direction means the FACTOR's intrinsic effect on {ticker} — NOT
   the price change itself.  A bearish event is bearish (direction < 0)
   even if the stock went UP because of other offsetting factors.

Return a JSON array.  Each factor:
- "name": short snake_case (e.g. "fed_rate_hike")
- "category": one of {categories}
- "direction": float -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float 0.0 to 1.0
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation

Magnitudes MUST sum to 1.0.  Include "market_noise" for unexplained variance.
Respond ONLY with valid JSON — no markdown, no commentary.
"""


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, model: str, api_key: str,
              temperature: float = 0.05, max_retries: int = 3) -> str:
    import litellm
    litellm.suppress_debug_info = True
    for attempt in range(max_retries + 1):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                api_key=api_key,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            # Don't retry on credit exhaustion — it won't help
            if "402" in err_str or "credits" in err_str.lower():
                raise
            if attempt == max_retries:
                raise
            wait = 5 * (2 ** attempt)
            print(f"      ⟳ Retry {attempt+1}/{max_retries} after {type(e).__name__} — waiting {wait}s")
            time.sleep(wait)


def _parse_factors(raw: str) -> List[Dict]:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        cleaned = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
        factors = json.loads(cleaned)
        out = []
        for f in factors:
            out.append({
                "name": str(f.get("name", "unknown")),
                "direction": float(f.get("direction", 0)),
                "magnitude": float(f.get("magnitude", 0.5)),
                "category": str(f.get("category", "unknown")),
                "description": str(f.get("description", "")),
            })
        return out
    except Exception:
        return []


def _get_news_context(ticker, start, end,
                      rag_dir="cache/rag_store", max_articles=50):
    try:
        from sde_causal_generator.news_rag import NewsRAGManager
        rag = NewsRAGManager(persist_dir=rag_dir)
        ctx = rag.get_context(ticker=ticker, start_date=start,
                              end_date=end, max_articles=max_articles)
        if ctx and "[No news articles found" not in ctx:
            return ctx
    except Exception:
        pass
    return ""


def _tokens(name: str) -> set:
    _STOP = frozenset({
        "the", "a", "an", "of", "in", "on", "to", "for", "and", "is",
        "market", "stock", "price", "impact", "effect", "factor", "data",
        "change", "movement", "related", "general", "overall", "aapl",
        "apple", "inc", "2018", "2019", "2020", "2021", "2022", "2023",
    })
    return set(name.lower().replace("-", "_").split("_")) - _STOP


def _factor_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ══════════════════════════════════════════════════════════════════════
# Fleiss' kappa
# ══════════════════════════════════════════════════════════════════════

def _fleiss_kappa(ratings: List[List[str]], categories: List[str]) -> float:
    n_items = len(ratings)
    n_raters = len(ratings[0]) if ratings else 0
    k = len(categories)
    if n_items == 0 or n_raters < 2:
        return float("nan")
    cat_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_items, k))
    for i, item_ratings in enumerate(ratings):
        for r in item_ratings:
            if r in cat_idx:
                counts[i, cat_idx[r]] += 1
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = float(np.sum(p_j ** 2))
    P_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(np.mean(P_i))
    if abs(1 - P_e) < 1e-10:
        return 1.0 if P_bar >= 1.0 - 1e-10 else 0.0
    return (P_bar - P_e) / (1 - P_e)


# ══════════════════════════════════════════════════════════════════════
# Pairwise concordance computation
# ══════════════════════════════════════════════════════════════════════

def compute_pairwise_concordance(
    all_window_factors: List[Dict[str, List[Dict]]],
    model_names: List[str],
) -> Tuple[np.ndarray, float]:
    """Compute pairwise directional agreement matrix and Fleiss' κ.

    Parameters
    ----------
    all_window_factors : list of dicts, one per window.
        Each dict maps model_name → list of factor dicts.
    model_names : list of model labels.

    Returns
    -------
    concordance : (n_models, n_models) agreement rate matrix.
    kappa : Fleiss' κ across all models.
    """
    n = len(model_names)
    agree_count = np.zeros((n, n))
    total_count = np.zeros((n, n))

    # For Fleiss' κ: each "item" = (window, factor_cluster), raters = models
    fleiss_ratings = []

    for wf in all_window_factors:
        # Cluster factors across models for this window
        clusters = _cluster_factors_for_concordance(wf, model_names)

        for cluster in clusters:
            # Collect direction classifications per model
            dir_class = {}  # model → "bullish" | "bearish" | "neutral"
            for mn in model_names:
                if mn in cluster:
                    d = cluster[mn]["direction"]
                    if d > 0.15:
                        dir_class[mn] = "bullish"
                    elif d < -0.15:
                        dir_class[mn] = "bearish"
                    else:
                        dir_class[mn] = "neutral"

            # Pairwise agreement
            present = [mn for mn in model_names if mn in dir_class]
            for i_idx, mi in enumerate(model_names):
                for j_idx, mj in enumerate(model_names):
                    if mi in dir_class and mj in dir_class:
                        total_count[i_idx, j_idx] += 1
                        if dir_class[mi] == dir_class[mj]:
                            agree_count[i_idx, j_idx] += 1

            # Fleiss' κ rating (need ≥2 models rating this item)
            if len(present) >= 2:
                row = []
                for mn in model_names:
                    if mn in dir_class:
                        row.append(dir_class[mn])
                    else:
                        row.append("neutral")  # padding
                fleiss_ratings.append(row)

    # Compute agreement rates
    concordance = np.where(
        total_count > 0,
        agree_count / total_count,
        0.0,
    )

    # Fleiss' κ
    kappa = _fleiss_kappa(
        fleiss_ratings,
        ["bullish", "bearish", "neutral"],
    )

    return concordance, kappa


def _cluster_factors_for_concordance(
    window_factors: Dict[str, List[Dict]],
    model_names: List[str],
    sim_threshold: float = 0.25,
) -> List[Dict[str, Dict]]:
    """Cluster factors across models for a single window.

    Returns list of clusters, each mapping model_name → factor dict.
    """
    pairs = []
    for mn in model_names:
        for f in window_factors.get(mn, []):
            if "noise" in f["name"] or "consolidation" in f["name"]:
                continue
            pairs.append((mn, f))

    if not pairs:
        return []

    clusters: List[Dict[str, Dict]] = []
    used = set()

    for i, (mi, fi) in enumerate(pairs):
        if i in used:
            continue
        cluster = {mi: fi}
        used.add(i)
        for j in range(i + 1, len(pairs)):
            if j in used:
                continue
            mj, fj = pairs[j]
            if mj in cluster:
                continue
            sim = _factor_similarity(fi["name"], fj["name"])
            same_cat = fi["category"] == fj["category"]
            if sim >= sim_threshold or (same_cat and sim >= 0.15):
                cluster[mj] = fj
                used.add(j)
        clusters.append(cluster)

    return clusters


# ══════════════════════════════════════════════════════════════════════
# Run a single condition (B or C)
# ══════════════════════════════════════════════════════════════════════

def run_condition(
    condition: str,
    windows: List[Dict],
    ticker: str,
    models: List[Tuple[str, str]],
    api_key: str,
    delay: float = 0.5,
    output_dir: str = "results/ablation_llm_test",
) -> List[Dict[str, List[Dict]]]:
    """Run LLM calls for one ablation condition.

    Returns list of dicts (one per window), each mapping model_name →
    list of factor dicts.
    """
    # Check for checkpoint
    ckpt_path = os.path.join(output_dir, f"checkpoint_{condition}.json")
    results = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  Resuming from checkpoint: {start_idx}/{len(windows)} windows done")

    total_calls = 0
    parse_failures = 0

    for wi, window in enumerate(windows):
        if wi < start_idx:
            continue
        start = window["start_date"]
        end = window["end_date"]
        real_pct = window["price_change_pct"]
        real_open = 100.0
        real_close = real_open * (1 + real_pct / 100)

        # Get RAG context
        news_ctx = _get_news_context(ticker, start, end)
        ctx_str = news_ctx if news_ctx else "(No news articles available for this period.)"

        # Build prompt based on condition
        if condition == "rag_only":
            prompt = _PROMPT_RAG_ONLY.format(
                ticker=ticker, start_date=start, end_date=end,
                news_context=ctx_str,
                categories=FACTOR_CATEGORIES,
            )
        elif condition == "price_only":
            prompt = _PROMPT_PRICE_ONLY.format(
                ticker=ticker, start_date=start, end_date=end,
                open_price=f"{real_open:.2f}",
                close_price=f"{real_close:.2f}",
                price_change_pct=real_pct,
                categories=FACTOR_CATEGORIES,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        window_factors: Dict[str, List[Dict]] = {}
        for model_label, model_id in models:
            try:
                raw = _call_llm(prompt, model_id, api_key)
                factors = _parse_factors(raw)
                if factors:
                    window_factors[model_label] = factors
                else:
                    parse_failures += 1
                total_calls += 1
            except Exception as e:
                print(f"    ERROR {model_label}: {e}")
                total_calls += 1
                parse_failures += 1
            time.sleep(delay)

        results.append(window_factors)
        n_models = len(window_factors)
        print(f"  [{condition}] [{wi+1}/{len(windows)}] {start}→{end} "
              f"pct={real_pct:+.1f}%  models={n_models}/5")

        # Checkpoint after each window
        with open(ckpt_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"  [{condition}] Done: {total_calls} calls, "
          f"{parse_failures} parse failures")
    return results


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_heatmap(
    concordance: np.ndarray,
    kappa: float,
    model_labels: List[str],
    title: str,
    output_path: str,
):
    """Generate a single pairwise concordance heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n = len(model_labels)

    # Custom green colormap matching the reference image
    colors = ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354",
              "#006d2c", "#00441b"]
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(concordance, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    # Determine κ interpretation
    if kappa >= 0.81:
        kappa_label = "Almost perfect"
    elif kappa >= 0.61:
        kappa_label = "Substantial"
    elif kappa >= 0.41:
        kappa_label = "Moderate"
    elif kappa >= 0.21:
        kappa_label = "Fair"
    else:
        kappa_label = "Poor"

    ax.set_title(
        f"{title}\n(Fleiss' κ = {kappa:.3f} — {kappa_label})",
        fontsize=14, fontweight="bold", pad=15,
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=11)

    # Annotate cells with percentage
    for i in range(n):
        for j in range(n):
            val = concordance[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val*100:.0f}%",
                    ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Agreement Rate")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined(
    results: Dict[str, Tuple[np.ndarray, float]],
    model_labels: List[str],
    output_path: str,
    ticker: str = "AAPL",
):
    """Generate a combined figure with all 3 heatmaps side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354",
              "#006d2c", "#00441b"]
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

    conditions = [
        ("A", "Price + RAG", "price_and_rag"),
        ("B", "RAG Only", "rag_only"),
        ("C", "Price Only", "price_only"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    fig.suptitle(
        f"LLM Ablation Test — Pairwise Directional Concordance — {ticker}",
        fontsize=16, fontweight="bold", y=1.02,
    )

    n = len(model_labels)

    for idx, (label, display, key) in enumerate(conditions):
        ax = axes[idx]
        concordance, kappa = results[key]

        if kappa >= 0.81:
            kappa_label = "Almost perfect"
        elif kappa >= 0.61:
            kappa_label = "Substantial"
        elif kappa >= 0.41:
            kappa_label = "Moderate"
        elif kappa >= 0.21:
            kappa_label = "Fair"
        else:
            kappa_label = "Poor"

        im = ax.imshow(concordance, cmap=cmap, vmin=0.0, vmax=1.0,
                        aspect="equal")

        ax.set_title(
            f"({label}) {display}\nFleiss' κ = {kappa:.3f} — {kappa_label}",
            fontsize=13, fontweight="bold", pad=10,
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(model_labels, fontsize=10)

        for i in range(n):
            for j in range(n):
                val = concordance[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val*100:.0f}%",
                        ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

    # Single colorbar on the right
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                         label="Agreement Rate", pad=0.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Ablation Test — Factor Extraction Sensitivity")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--output-dir", default="results/ablation_llm_test")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between LLM calls (seconds)")
    parser.add_argument("--cross-llm-results",
                        default="results/cross_llm_semantic_test/cross_llm_semantic_results.json",
                        help="Path to existing cross-LLM results (condition A)")
    parser.add_argument("--analyses",
                        default=None,
                        help="Path to analyses.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    model_names = [m[0] for m in MODELS]
    short_labels = [MODEL_SHORT.get(m, m) for m in model_names]

    # ── Load condition A from existing cross-LLM results ────────────
    print("=" * 70)
    print("LLM ABLATION TEST — FACTOR EXTRACTION INPUT SENSITIVITY")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Models: {len(MODELS)} ({', '.join(model_names)})")
    print()

    print("─── Condition A: Price + RAG (reusing cross-LLM results) ───")
    if not os.path.exists(args.cross_llm_results):
        print(f"ERROR: {args.cross_llm_results} not found")
        sys.exit(1)

    with open(args.cross_llm_results) as f:
        cross_llm = json.load(f)

    # Extract per-window factors for condition A
    cond_a_factors = []
    windows_info = []
    for w in cross_llm["per_window"]:
        cond_a_factors.append(w["real_factors_by_model"])
        parts = w["window"].split(" → ")
        windows_info.append({
            "start_date": parts[0].strip(),
            "end_date": parts[1].strip(),
            "price_change_pct": w["real_pct"],
        })

    print(f"  Loaded {len(cond_a_factors)} windows from existing results")
    print(f"  (zero new LLM calls needed)")

    # ── Run condition B: RAG only ───────────────────────────────────
    print()
    print("─── Condition B: RAG Only (new LLM calls) ───")
    cond_b_factors = run_condition(
        condition="rag_only",
        windows=windows_info,
        ticker=args.ticker,
        models=MODELS,
        api_key=api_key,
        delay=args.delay,
        output_dir=args.output_dir,
    )

    # ── Run condition C: Price only ─────────────────────────────────
    print()
    print("─── Condition C: Price Only (new LLM calls) ───")
    cond_c_factors = run_condition(
        condition="price_only",
        windows=windows_info,
        ticker=args.ticker,
        models=MODELS,
        api_key=api_key,
        delay=args.delay,
        output_dir=args.output_dir,
    )

    # ── Compute concordance matrices ────────────────────────────────
    print()
    print("─── Computing concordance matrices ───")

    conc_a, kappa_a = compute_pairwise_concordance(cond_a_factors, model_names)
    conc_b, kappa_b = compute_pairwise_concordance(cond_b_factors, model_names)
    conc_c, kappa_c = compute_pairwise_concordance(cond_c_factors, model_names)

    print(f"  (A) Price + RAG:  κ = {kappa_a:.3f}")
    print(f"  (B) RAG only:     κ = {kappa_b:.3f}")
    print(f"  (C) Price only:   κ = {kappa_c:.3f}")

    # ── Save results ────────────────────────────────────────────────
    results_data = {
        "experiment": "LLM Ablation Test — Factor Extraction Input Sensitivity",
        "ticker": args.ticker,
        "models": [{"label": m[0], "id": m[1]} for m in MODELS],
        "n_windows": len(windows_info),
        "conditions": {
            "price_and_rag": {
                "description": "Full pipeline: price data + RAG news context",
                "source": "reused from cross-LLM semantic test",
                "fleiss_kappa": round(float(kappa_a), 4),
                "concordance_matrix": conc_a.tolist(),
            },
            "rag_only": {
                "description": "RAG news context only, no price data",
                "source": "new LLM calls",
                "fleiss_kappa": round(float(kappa_b), 4),
                "concordance_matrix": conc_b.tolist(),
            },
            "price_only": {
                "description": "Price data only, no news context",
                "source": "new LLM calls",
                "fleiss_kappa": round(float(kappa_c), 4),
                "concordance_matrix": conc_c.tolist(),
            },
        },
        "windows": windows_info,
        "raw_factors": {
            "price_and_rag": [
                {mn: fs for mn, fs in wf.items()} for wf in cond_a_factors
            ],
            "rag_only": [
                {mn: fs for mn, fs in wf.items()} for wf in cond_b_factors
            ],
            "price_only": [
                {mn: fs for mn, fs in wf.items()} for wf in cond_c_factors
            ],
        },
    }

    out_json = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_json}")

    # ── Generate plots ──────────────────────────────────────────────
    print()
    print("─── Generating heatmaps ───")

    plot_heatmap(
        conc_a, kappa_a, short_labels,
        f"(A) Price + RAG — {args.ticker}",
        os.path.join(args.output_dir, "heatmap_A_price_and_rag.png"),
    )
    plot_heatmap(
        conc_b, kappa_b, short_labels,
        f"(B) RAG Only — {args.ticker}",
        os.path.join(args.output_dir, "heatmap_B_rag_only.png"),
    )
    plot_heatmap(
        conc_c, kappa_c, short_labels,
        f"(C) Price Only — {args.ticker}",
        os.path.join(args.output_dir, "heatmap_C_price_only.png"),
    )

    # Combined figure
    all_results = {
        "price_and_rag": (conc_a, kappa_a),
        "rag_only": (conc_b, kappa_b),
        "price_only": (conc_c, kappa_c),
    }
    plot_combined(
        all_results, short_labels,
        os.path.join(args.output_dir, "heatmap_combined.png"),
        ticker=args.ticker,
    )

    # Also save as PDF
    plot_combined(
        all_results, short_labels,
        os.path.join(args.output_dir, "heatmap_combined.pdf"),
        ticker=args.ticker,
    )

    # ── Print summary ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("LLM ABLATION TEST — SUMMARY")
    print("=" * 70)
    print(f"  Condition A (Price + RAG):  Fleiss' κ = {kappa_a:.3f}")
    print(f"  Condition B (RAG only):     Fleiss' κ = {kappa_b:.3f}")
    print(f"  Condition C (Price only):   Fleiss' κ = {kappa_c:.3f}")
    print()

    # Interpretation
    kappas = {"A": kappa_a, "B": kappa_b, "C": kappa_c}
    best = max(kappas, key=kappas.get)
    worst = min(kappas, key=kappas.get)

    if kappas["B"] >= kappas["C"]:
        print("  → RAG-grounded factors show HIGHER cross-LLM agreement than")
        print("    price-only factors. The news context provides a shared")
        print("    factual anchor that improves inter-model consistency.")
    else:
        print("  → Price-only factors show HIGHER cross-LLM agreement than")
        print("    RAG-only factors. LLMs converge more when given explicit")
        print("    price signals.")

    if kappas["A"] >= max(kappas["B"], kappas["C"]):
        print("  → The full pipeline (A) achieves the HIGHEST concordance,")
        print("    confirming that combining price + RAG is optimal.")
    else:
        print(f"  → Condition {best} achieves the highest concordance.")

    diff_ab = abs(kappa_a - kappa_b)
    if diff_ab < 0.05:
        print("  → Minimal difference between A and B suggests price data")
        print("    barely affects factor identification when RAG is present.")

    print()
    print(f"  Output: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
