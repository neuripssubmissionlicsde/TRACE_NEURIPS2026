#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual Price Injection — LLM Semantic Direction Test.

Tests whether the LLM analyses factors **semantically** (based on the
causal nature of the event) or simply **mirrors price direction** (the
sign of price_change_pct in the prompt).

Design
------
For N time-windows sampled from AAPL (2018–2023):

* **Group A (control)**: LLM receives the real price_change_pct.
  Already cached from the pipeline run → zero cost.
* **Group B (counterfactual)**: LLM receives the **inverted** price
  (price_change_pct × −1, plus swapped open/close prices).  The RAG
  news context stays IDENTICAL — only numeric price data changes.

Metrics
-------
For each factor that appears in **both** A and B for the same window:
  - Pearson ρ between direction_A and direction_B
  - Sign agreement rate: fraction where sign(d_A) == sign(d_B)
  - Δ_direction = |d_A − d_B|

If the LLM is semantically grounded:
  - ρ(d_A, d_B) ≈ 1   (directions track the factor, not the price)
  - sign agreement ≈ 1
  - Δ ≈ 0

If the LLM mirrors price:
  - ρ ≈ −1  (directions flip when price flips)
  - sign agreement ≈ 0

Results saved to:
  results/semantic_direction_test/
    semantic_direction_results.json
    fig_semantic_direction.png / .pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Inline definition to avoid heavy package import chain (pandas→pyarrow→numpy2 conflict)
FACTOR_CATEGORIES = [
    "macroeconomic",
    "geopolitical",
    "sector_specific",
    "company_specific",
    "market_sentiment",
    "technical",
    "liquidity",
]


# ══════════════════════════════════════════════════════════════════════
# LLM calling (reuses the same prompts as factor_extractor)
# ══════════════════════════════════════════════════════════════════════

def _call_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.05,
) -> str:
    """Call LLM via litellm (OpenRouter)."""
    import litellm
    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        api_key=api_key,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_factors(raw_json: str) -> List[Dict]:
    """Parse LLM JSON into list of factor dicts.  Returns [] on failure."""
    import re
    try:
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        cleaned = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
        factors = json.loads(cleaned)
        return [
            {
                "name": f["name"],
                "direction": float(f["direction"]),
                "magnitude": float(f.get("magnitude", 0.5)),
                "category": f.get("category", "unknown"),
            }
            for f in factors
        ]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════
# Prompt (simplified — no RAG to isolate the price-direction effect)
# ══════════════════════════════════════════════════════════════════════

_COUNTERFACTUAL_PROMPT = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Your goal is to identify the specific
factors that CAUSED price movements — not factors that merely coincided.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

PRICE DATA:
- Opening price: {open_price}
- Closing price: {close_price}
- Price change: {price_change_pct:.2f}%

{news_context}

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. If no relevant news explains the price movement, attribute it to
   "market_noise" with high magnitude.
3. Do NOT reference any events that are NOT in the provided news context.
4. Do NOT use knowledge of what happened AFTER {end_date}.
5. For each factor, cite the article number [N] that supports it.
6. Keep magnitude and confidence SEPARATE: magnitude = how much market
   impact; confidence = how certain you are of the causal link.
7. direction means the FACTOR's intrinsic effect on the stock — NOT
   the price change itself.  A bearish event is bearish (direction < 0)
   even if the stock went UP because of other offsetting factors.

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier (e.g., "fed_rate_hike")
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation citing specific evidence

The magnitudes across all factors MUST sum to 1.0.
Include a "market_noise" factor for truly unexplained random variance.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""


# ══════════════════════════════════════════════════════════════════════
# Window selection
# ══════════════════════════════════════════════════════════════════════

def select_windows(
    analyses_path: str,
    n_windows: int = 50,
    min_abs_pct: float = 2.0,
) -> List[Dict]:
    """Select windows with significant price moves, stratified by sign.

    Returns list of dicts with keys:
      start_date, end_date, price_change_pct, factors (list[dict])
    """
    with open(analyses_path) as f:
        analyses = json.load(f)

    # Filter to windows with significant moves
    eligible = [
        a for a in analyses
        if abs(a["price_change_pct"]) >= min_abs_pct
        and len([f for f in a["factors"]
                 if "noise" not in f["name"]
                 and "consolidation" not in f["name"]]) >= 2
    ]

    # Stratified sampling: half bullish, half bearish
    bullish = sorted(
        [a for a in eligible if a["price_change_pct"] > 0],
        key=lambda a: -abs(a["price_change_pct"]),
    )
    bearish = sorted(
        [a for a in eligible if a["price_change_pct"] < 0],
        key=lambda a: -abs(a["price_change_pct"]),
    )

    half = n_windows // 2
    selected = bullish[:half] + bearish[:half]

    # If not enough bearish/bullish, fill from the other
    if len(selected) < n_windows:
        remaining = [a for a in eligible if a not in selected]
        selected += remaining[: n_windows - len(selected)]

    print(f"Selected {len(selected)} windows "
          f"({sum(1 for w in selected if w['price_change_pct'] > 0)} bullish, "
          f"{sum(1 for w in selected if w['price_change_pct'] < 0)} bearish)")

    return selected[:n_windows]


# ══════════════════════════════════════════════════════════════════════
# News context retrieval
# ══════════════════════════════════════════════════════════════════════

def _get_news_context(
    ticker: str,
    start_date: str,
    end_date: str,
    rag_persist_dir: str = "cache/rag_store",
    max_articles: int = 50,
    content_limit: int = 500,
) -> str:
    """Retrieve cached RAG context for a window (same as pipeline)."""
    try:
        from sde_causal_generator.news_rag import NewsRAGManager
        rag = NewsRAGManager(persist_dir=rag_persist_dir)
        ctx = rag.get_context(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles,
        )
        if ctx and "[No news articles found" not in ctx:
            return ctx
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════
# Run experiment
# ══════════════════════════════════════════════════════════════════════

def run_experiment(
    windows: List[Dict],
    ticker: str = "AAPL",
    model: str = "openrouter/google/gemma-3-27b-it",
    api_key: str = "",
    n_samples: int = 3,
    output_dir: str = "results/semantic_direction_test",
    delay: float = 1.0,
) -> Dict:
    """Run the counterfactual price injection experiment.

    For each window:
    - Group A = existing analysis from the pipeline (real price)
    - Group B = LLM call with inverted price_change_pct

    Returns experiment results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    results = []
    total_calls = 0
    parse_failures = 0

    for wi, window in enumerate(windows):
        start = window["start_date"]
        end = window["end_date"]
        real_pct = window["price_change_pct"]

        # Group A: factors from the pipeline run (already analysed)
        control_factors = [
            {
                "name": f["name"],
                "direction": f["direction"],
                "magnitude": f.get("magnitude", 0.5),
                "category": f.get("category", "unknown"),
            }
            for f in window["factors"]
            if "noise" not in f["name"] and "consolidation" not in f["name"]
        ]

        if len(control_factors) < 1:
            continue

        # ── Build counterfactual prompt ─────────────────────────────
        # Invert the price and swap open/close
        fake_pct = -real_pct

        # Get RAG news context (same for both groups)
        news_ctx = _get_news_context(ticker, start, end)

        # Simulate swapped prices (approximate)
        # If real open=100, close=110 (pct=+10%), fake should be open=110, close=100 (pct=-9.09%)
        # We use the already-computed flip: just negate the percentage
        # and swap open/close labels
        real_open = 100.0  # placeholder — the LLM prompt is about *direction*, not exact values
        real_close = real_open * (1 + real_pct / 100)
        fake_open = real_close  # swap
        fake_close = real_open

        prompt = _COUNTERFACTUAL_PROMPT.format(
            ticker=ticker,
            start_date=start,
            end_date=end,
            open_price=f"{fake_open:.2f}",
            close_price=f"{fake_close:.2f}",
            price_change_pct=fake_pct,
            news_context=news_ctx if news_ctx else "(No news articles available for this period.)",
            categories=FACTOR_CATEGORIES,
        )

        # ── Group B: counterfactual LLM calls ──────────────────────
        counterfactual_factors_all: Dict[str, List[Dict]] = {}
        n_ok = 0

        for s in range(n_samples):
            try:
                raw = _call_llm(prompt, model, api_key)
                factors = _parse_factors(raw)
                if factors:
                    n_ok += 1
                    for f in factors:
                        counterfactual_factors_all.setdefault(f["name"], []).append(f)
                else:
                    parse_failures += 1
                total_calls += 1
            except Exception as e:
                print(f"    ERROR calling LLM: {e}")
                total_calls += 1
                parse_failures += 1

            time.sleep(delay)

        # Average across samples
        counterfactual_factors = {}
        for name, flist in counterfactual_factors_all.items():
            if len(flist) >= max(1, n_samples / 2):
                counterfactual_factors[name] = {
                    "name": name,
                    "direction": float(np.mean([f["direction"] for f in flist])),
                    "magnitude": float(np.mean([f["magnitude"] for f in flist])),
                    "category": flist[0]["category"],
                }

        # ── Compare: match factors between A and B ──────────────────
        # Stopwords that add no semantic value for matching
        _STOP = {"the", "a", "an", "of", "in", "on", "to", "for", "and",
                 "market", "stock", "price", "impact", "effect", "factor"}

        def _tokens(name: str) -> set:
            return set(name.lower().split("_")) - _STOP

        control_by_name = {f["name"]: f for f in control_factors}
        matched = []
        used_cf = set()  # prevent double-matching

        # Pass 1: exact match
        for name, ctrl in control_by_name.items():
            if name in counterfactual_factors and name not in used_cf:
                cf = counterfactual_factors[name]
                matched.append({
                    "name": name,
                    "direction_control": ctrl["direction"],
                    "direction_counterfactual": cf["direction"],
                    "magnitude_control": ctrl["magnitude"],
                    "magnitude_counterfactual": cf["magnitude"],
                    "match_type": "exact",
                })
                used_cf.add(name)

        # Pass 2: best token-overlap match (greedy, 1:1)
        unmatched_ctrl = [n for n in control_by_name if n not in
                          {m["name"].split(" ≈ ")[0] for m in matched}
                          and n not in {m["name"] for m in matched}]
        avail_cf = {n: f for n, f in counterfactual_factors.items()
                    if n not in used_cf}

        for ctrl_name in unmatched_ctrl:
            tokens_a = _tokens(ctrl_name)
            if not tokens_a:
                continue
            best_score = 0.0
            best_cf_name = None
            for cf_name in avail_cf:
                tokens_b = _tokens(cf_name)
                if not tokens_b:
                    continue
                common = tokens_a & tokens_b
                # Jaccard similarity
                score = len(common) / len(tokens_a | tokens_b) if (tokens_a | tokens_b) else 0
                if score > best_score:
                    best_score = score
                    best_cf_name = cf_name
            # Accept if ≥1 meaningful common token
            if best_cf_name and best_score >= 0.15:
                cf = avail_cf[best_cf_name]
                ctrl = control_by_name[ctrl_name]
                matched.append({
                    "name": f"{ctrl_name} ≈ {best_cf_name}",
                    "direction_control": ctrl["direction"],
                    "direction_counterfactual": cf["direction"],
                    "magnitude_control": ctrl["magnitude"],
                    "magnitude_counterfactual": cf["magnitude"],
                    "match_type": f"fuzzy({best_score:.2f})",
                })
                used_cf.add(best_cf_name)
                del avail_cf[best_cf_name]

        # Pass 3: category-based fallback (match by same category)
        still_unmatched = [n for n in control_by_name if n not in
                           {m["name"].split(" ≈ ")[0] for m in matched}
                           and n not in {m["name"] for m in matched}]
        for ctrl_name in still_unmatched:
            ctrl = control_by_name[ctrl_name]
            ctrl_cat = ctrl.get("category", "")
            for cf_name, cf in list(avail_cf.items()):
                if cf.get("category", "") == ctrl_cat and ctrl_cat:
                    matched.append({
                        "name": f"{ctrl_name} ≈ {cf_name}",
                        "direction_control": ctrl["direction"],
                        "direction_counterfactual": cf["direction"],
                        "magnitude_control": ctrl["magnitude"],
                        "magnitude_counterfactual": cf["magnitude"],
                        "match_type": f"category({ctrl_cat})",
                    })
                    used_cf.add(cf_name)
                    del avail_cf[cf_name]
                    break

        result = {
            "window": f"{start} → {end}",
            "real_price_change_pct": real_pct,
            "fake_price_change_pct": fake_pct,
            "n_control_factors": len(control_factors),
            "n_counterfactual_factors": len(counterfactual_factors),
            "n_matched": len(matched),
            "matched_factors": matched,
            "n_llm_parsed_ok": n_ok,
            "control_factor_names": [f["name"] for f in control_factors],
            "counterfactual_factor_names": list(counterfactual_factors.keys()),
        }
        results.append(result)

        # Progress
        n_matched_total = sum(r["n_matched"] for r in results)
        print(f"  [{wi+1}/{len(windows)}] {start}→{end}  "
              f"pct={real_pct:+.1f}%→{fake_pct:+.1f}%  "
              f"matched={len(matched)}  total_matched={n_matched_total}")

    # ══════════════════════════════════════════════════════════════════
    # Aggregate metrics
    # ══════════════════════════════════════════════════════════════════

    all_dir_control = []
    all_dir_counterfactual = []
    sign_agreements = []
    direction_deltas = []

    for r in results:
        for m in r["matched_factors"]:
            dc = m["direction_control"]
            df_ = m["direction_counterfactual"]
            all_dir_control.append(dc)
            all_dir_counterfactual.append(df_)
            # Skip near-zero (neutral) directions
            if abs(dc) > 0.1 and abs(df_) > 0.1:
                sign_agreements.append(
                    1.0 if np.sign(dc) == np.sign(df_) else 0.0
                )
            direction_deltas.append(abs(dc - df_))

    # Pearson correlation
    pearson_r = float("nan")
    if len(all_dir_control) >= 3:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(all_dir_control, all_dir_counterfactual)
        spearman_r, spearman_p = spearmanr(all_dir_control, all_dir_counterfactual)
    else:
        pearson_p = float("nan")
        spearman_r = float("nan")
        spearman_p = float("nan")

    sign_agree_rate = float(np.mean(sign_agreements)) if sign_agreements else float("nan")
    mean_delta = float(np.mean(direction_deltas)) if direction_deltas else float("nan")
    median_delta = float(np.median(direction_deltas)) if direction_deltas else float("nan")

    # ── Classification ─────────────────────────────────────────────
    # Factor-level: classify each factor as "semantic", "price-following", or "ambiguous"
    semantic_count = 0
    price_following_count = 0
    ambiguous_count = 0
    factor_classifications = []

    for r in results:
        real_pct = r["real_price_change_pct"]
        for m in r["matched_factors"]:
            dc = m["direction_control"]
            df_ = m["direction_counterfactual"]
            if abs(dc) < 0.1 or abs(df_) < 0.1:
                classification = "neutral"
                ambiguous_count += 1
            elif np.sign(dc) == np.sign(df_):
                classification = "semantic"
                semantic_count += 1
            else:
                # Direction flipped — but did it flip to match the fake price?
                if np.sign(df_) == np.sign(r["fake_price_change_pct"]):
                    classification = "price_following"
                    price_following_count += 1
                else:
                    classification = "ambiguous"
                    ambiguous_count += 1
            factor_classifications.append({
                "factor": m["name"],
                "window": r["window"],
                "direction_control": dc,
                "direction_counterfactual": df_,
                "real_pct": real_pct,
                "fake_pct": r["fake_price_change_pct"],
                "classification": classification,
            })

    total_classified = semantic_count + price_following_count + ambiguous_count
    semantic_pct = 100 * semantic_count / total_classified if total_classified > 0 else 0
    price_following_pct = 100 * price_following_count / total_classified if total_classified > 0 else 0

    # ── Build summary ──────────────────────────────────────────────
    summary = {
        "experiment": "Counterfactual Price Injection — Semantic Direction Test",
        "ticker": ticker,
        "model": model,
        "n_windows": len(windows),
        "n_windows_with_matches": sum(1 for r in results if r["n_matched"] > 0),
        "total_matched_factors": len(all_dir_control),
        "total_llm_calls": total_calls,
        "total_parse_failures": parse_failures,
        "n_samples_per_window": n_samples,
        "metrics": {
            "pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else None,
            "pearson_p": round(pearson_p, 6) if not np.isnan(pearson_p) else None,
            "spearman_r": round(spearman_r, 4) if not np.isnan(spearman_r) else None,
            "spearman_p": round(spearman_p, 6) if not np.isnan(spearman_p) else None,
            "sign_agreement_rate": round(sign_agree_rate, 4) if not np.isnan(sign_agree_rate) else None,
            "mean_direction_delta": round(mean_delta, 4) if not np.isnan(mean_delta) else None,
            "median_direction_delta": round(median_delta, 4) if not np.isnan(median_delta) else None,
        },
        "classification": {
            "semantic": semantic_count,
            "price_following": price_following_count,
            "ambiguous": ambiguous_count,
            "total": total_classified,
            "semantic_pct": round(semantic_pct, 1),
            "price_following_pct": round(price_following_pct, 1),
        },
        "interpretation": _interpret(pearson_r, sign_agree_rate, semantic_pct),
        "per_window": results,
        "factor_classifications": factor_classifications,
    }

    # Save results
    out_path = os.path.join(output_dir, "semantic_direction_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SEMANTIC DIRECTION TEST — RESULTS")
    print("=" * 60)
    print(f"Windows analysed:          {len(windows)}")
    print(f"Windows with matched factors: {sum(1 for r in results if r['n_matched'] > 0)}")
    print(f"Total matched factors:     {len(all_dir_control)}")
    print(f"LLM calls:                 {total_calls} ({parse_failures} parse failures)")
    print()
    print("─── Correlation metrics ───")
    print(f"Pearson ρ(d_A, d_B):       {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"Spearman ρ(d_A, d_B):      {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"Sign agreement rate:       {sign_agree_rate:.1%}")
    print(f"Mean |Δ direction|:        {mean_delta:.4f}")
    print(f"Median |Δ direction|:      {median_delta:.4f}")
    print()
    print("─── Classification ───")
    print(f"Semantic (direction stable):   {semantic_count:3d} ({semantic_pct:.1f}%)")
    print(f"Price-following (dir flipped): {price_following_count:3d} ({price_following_pct:.1f}%)")
    print(f"Ambiguous/neutral:             {ambiguous_count:3d}")
    print()
    print(f"Interpretation: {_interpret(pearson_r, sign_agree_rate, semantic_pct)}")
    print("=" * 60)

    return summary


def _interpret(pearson_r: float, sign_agree: float, semantic_pct: float) -> str:
    """Human-readable interpretation of the results."""
    if np.isnan(pearson_r):
        return "Insufficient data for interpretation."

    if pearson_r > 0.6 and semantic_pct > 70:
        return (
            "STRONG EVIDENCE for semantic analysis. The LLM maintains consistent "
            "factor directions even when price data is inverted, demonstrating that "
            "direction assignments are grounded in causal factor semantics, not "
            "price-following."
        )
    elif pearson_r > 0.3 and semantic_pct > 50:
        return (
            "MODERATE EVIDENCE for semantic analysis. The LLM mostly preserves "
            "factor directions under counterfactual prices, though some factors "
            "show sensitivity to the price signal. The dominant behaviour is semantic."
        )
    elif pearson_r < -0.3 and semantic_pct < 30:
        return (
            "EVIDENCE for price-following. The LLM reverses factor directions "
            "when price data is inverted, suggesting that direction assignments "
            "are primarily driven by the price signal, not factor semantics."
        )
    else:
        return (
            f"MIXED results. Pearson ρ={pearson_r:.3f}, semantic_pct={semantic_pct:.0f}%. "
            "The LLM shows partial semantic grounding with some price sensitivity."
        )


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_results(summary: Dict, output_dir: str):
    """Generate publication-ready figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    classifications = summary.get("factor_classifications", [])
    if not classifications:
        print("No classifications to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Counterfactual Price Injection — LLM Semantic Direction Test",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Scatter direction_A vs direction_B ─────────────────
    ax = axes[0]
    dirs_a = [c["direction_control"] for c in classifications]
    dirs_b = [c["direction_counterfactual"] for c in classifications]
    colors = {
        "semantic": "#2ecc71",
        "price_following": "#e74c3c",
        "ambiguous": "#95a5a6",
        "neutral": "#bdc3c7",
    }
    for cls_type, color in colors.items():
        mask = [c["classification"] == cls_type for c in classifications]
        ax.scatter(
            [d for d, m in zip(dirs_a, mask) if m],
            [d for d, m in zip(dirs_b, mask) if m],
            c=color, alpha=0.6, s=30, label=cls_type, edgecolors="white", linewidths=0.5,
        )
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="d_A = d_B")
    ax.plot([-1, 1], [1, -1], "r:", alpha=0.3, label="d_A = -d_B")
    ax.set_xlabel("Direction (real price)", fontsize=11)
    ax.set_ylabel("Direction (inverted price)", fontsize=11)
    ax.set_title(f"Factor Directions\nPearson ρ = {summary['metrics']['pearson_r']}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Classification pie chart ───────────────────────────
    ax = axes[1]
    cls = summary["classification"]
    sizes = [cls["semantic"], cls["price_following"], cls["ambiguous"]]
    labels = [
        f"Semantic\n{cls['semantic']} ({cls['semantic_pct']:.0f}%)",
        f"Price-following\n{cls['price_following']} ({cls['price_following_pct']:.0f}%)",
        f"Ambiguous\n{cls['ambiguous']}",
    ]
    pie_colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    # Only plot non-zero slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, pie_colors) if s > 0]
    if non_zero:
        ax.pie(
            [x[0] for x in non_zero],
            labels=[x[1] for x in non_zero],
            colors=[x[2] for x in non_zero],
            autopct=None,
            startangle=90,
            textprops={"fontsize": 10},
        )
    ax.set_title("Factor Classification", fontsize=11)

    # ── Panel 3: Direction delta histogram ──────────────────────────
    ax = axes[2]
    deltas = [
        abs(c["direction_control"] - c["direction_counterfactual"])
        for c in classifications
        if abs(c["direction_control"]) > 0.1
    ]
    if deltas:
        ax.hist(deltas, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(deltas), color="red", linestyle="--", label=f"Mean = {np.mean(deltas):.3f}")
        ax.axvline(np.median(deltas), color="orange", linestyle="--", label=f"Median = {np.median(deltas):.3f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("|Δ direction|", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Direction Change Distribution", fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"fig_semantic_direction.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures saved to {output_dir}/fig_semantic_direction.{{png,pdf}}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual Price Injection — Semantic Direction Test",
    )
    parser.add_argument(
        "--ticker", default="AAPL", help="Ticker to test (default: AAPL)",
    )
    parser.add_argument(
        "--analyses", default=None,
        help="Path to analyses.json (default: results/multi_asset_2018_2023/{ticker}/analyses.json)",
    )
    parser.add_argument(
        "--n-windows", type=int, default=50,
        help="Number of windows to test (default: 50)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="LLM calls per window for counterfactual (default: 3)",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model (default: from config yaml)",
    )
    parser.add_argument(
        "--output-dir", default="results/semantic_direction_test",
        help="Output directory",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between LLM calls in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    # Load config for model
    if args.model is None:
        import yaml
        config_path = PROJECT_ROOT / "configs" / "causal_sde_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model = config.get("llm_model", "openrouter/google/gemma-3-27b-it")
    else:
        model = args.model

    # Analyses path
    if args.analyses is None:
        analyses_path = str(
            PROJECT_ROOT / "results" / "multi_asset_2018_2023" / args.ticker / "analyses.json"
        )
    else:
        analyses_path = args.analyses

    if not os.path.exists(analyses_path):
        print(f"ERROR: {analyses_path} not found. Run the pipeline first.")
        sys.exit(1)

    print("=" * 60)
    print("COUNTERFACTUAL PRICE INJECTION — SEMANTIC DIRECTION TEST")
    print("=" * 60)
    print(f"Ticker:     {args.ticker}")
    print(f"Model:      {model}")
    print(f"Windows:    {args.n_windows}")
    print(f"Samples:    {args.n_samples} per window")
    print(f"Analyses:   {analyses_path}")
    print(f"Output:     {args.output_dir}")
    print()

    # Select windows
    windows = select_windows(analyses_path, args.n_windows)
    if not windows:
        print("ERROR: No eligible windows found")
        sys.exit(1)

    # Run experiment
    summary = run_experiment(
        windows=windows,
        ticker=args.ticker,
        model=model,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        delay=args.delay,
    )

    # Generate plots
    plot_results(summary, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
