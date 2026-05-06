#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-LLM Counterfactual Price Injection — Semantic Direction Test.

Extends the single-model counterfactual test to **multiple LLMs**.
For each time window, every model receives:
  - **Real prompt**:          real price_change_pct + RAG news
  - **Counterfactual prompt**: inverted price_change_pct + same RAG news

Metrics (per model AND cross-model):
  - Per-model:  sign agreement, Pearson ρ, semantic%
  - Cross-model: Fleiss' κ on factor direction classification
  - Cross-model: inter-model concordance on counterfactual behaviour

If the LLMs are semantically grounded:
  - All models keep factor direction stable under price inversion
  - Cross-model κ on counterfactual direction ≈ cross-model κ on real direction

Results saved to:
  results/cross_llm_semantic_test/
    cross_llm_semantic_results.json
    fig_cross_llm_semantic.png / .pdf
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

# ── Models ──────────────────────────────────────────────────────────
MODELS = [
    ("Gemma-3-27B",       "openrouter/google/gemma-3-27b-it"),
    ("GPT-4o-mini",       "openrouter/openai/gpt-4o-mini"),
    ("Claude-3.5-Haiku",  "openrouter/anthropic/claude-3.5-haiku"),
    ("Mistral-Small-3.1", "openrouter/mistralai/mistral-small-3.1-24b-instruct"),
    ("Gemini-2.0-Flash",  "openrouter/google/gemini-2.0-flash-001"),
]

# ── Prompt ──────────────────────────────────────────────────────────
_PROMPT = """\
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

# ── Stopwords for factor-name matching ──────────────────────────────
_STOP = frozenset({
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "is",
    "market", "stock", "price", "impact", "effect", "factor", "data",
    "change", "movement", "related", "general", "overall", "aapl",
    "apple", "inc", "2018", "2019", "2020", "2021", "2022", "2023",
})


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, model: str, api_key: str, temperature: float = 0.05) -> str:
    import litellm
    litellm.suppress_debug_info = True
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        api_key=api_key,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


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


def _tokens(name: str) -> set:
    return set(name.lower().replace("-", "_").split("_")) - _STOP


def _factor_similarity(a: str, b: str) -> float:
    """Jaccard similarity on meaningful tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _get_news_context(ticker, start, end, rag_dir="cache/rag_store", max_articles=50):
    try:
        from sde_causal_generator.news_rag import NewsRAGManager
        rag = NewsRAGManager(persist_dir=rag_dir)
        ctx = rag.get_context(ticker=ticker, start_date=start, end_date=end,
                              max_articles=max_articles)
        if ctx and "[No news articles found" not in ctx:
            return ctx
    except Exception:
        pass
    return ""


def select_windows(analyses_path: str, n_windows: int = 20, min_abs_pct: float = 2.0):
    with open(analyses_path) as f:
        analyses = json.load(f)
    eligible = [
        a for a in analyses
        if abs(a["price_change_pct"]) >= min_abs_pct
        and len([f for f in a["factors"]
                 if "noise" not in f["name"] and "consolidation" not in f["name"]]) >= 2
    ]
    bullish = sorted([a for a in eligible if a["price_change_pct"] > 0],
                     key=lambda a: -abs(a["price_change_pct"]))
    bearish = sorted([a for a in eligible if a["price_change_pct"] < 0],
                     key=lambda a: -abs(a["price_change_pct"]))
    half = n_windows // 2
    selected = bullish[:half] + bearish[:half]
    if len(selected) < n_windows:
        remaining = [a for a in eligible if a not in selected]
        selected += remaining[:n_windows - len(selected)]
    print(f"Selected {len(selected)} windows "
          f"({sum(1 for w in selected if w['price_change_pct'] > 0)} bullish, "
          f"{sum(1 for w in selected if w['price_change_pct'] < 0)} bearish)")
    return selected[:n_windows]


# ════════════════════════════════════════════════════════════════════
# Semantic factor clustering across LLMs
# ════════════════════════════════════════════════════════════════════

def _cluster_factors(all_model_factors: Dict[str, List[Dict]],
                     sim_threshold: float = 0.25) -> List[Dict]:
    """Cluster semantically equivalent factors across multiple models.

    Returns list of clusters, each with:
      canonical_name, category, per_model (dict model→{direction, magnitude, ...})
    """
    # Collect all (model, factor) pairs
    pairs = []
    for model_name, factors in all_model_factors.items():
        for f in factors:
            if "noise" in f["name"] or "consolidation" in f["name"]:
                continue
            pairs.append((model_name, f))

    if not pairs:
        return []

    # Greedy clustering by name similarity + category
    clusters: List[Dict] = []
    used = set()

    for i, (model_i, fi) in enumerate(pairs):
        if i in used:
            continue
        cluster = {
            "canonical_name": fi["name"],
            "category": fi["category"],
            "per_model": {model_i: fi},
        }
        used.add(i)

        for j in range(i + 1, len(pairs)):
            if j in used:
                continue
            model_j, fj = pairs[j]
            if model_j in cluster["per_model"]:
                continue  # one factor per model per cluster
            # Similarity check: name tokens OR same category + partial overlap
            sim = _factor_similarity(fi["name"], fj["name"])
            same_cat = fi["category"] == fj["category"]
            if sim >= sim_threshold or (same_cat and sim >= 0.15):
                cluster["per_model"][model_j] = fj
                used.add(j)

        clusters.append(cluster)

    # Sort: clusters with more models first
    clusters.sort(key=lambda c: -len(c["per_model"]))
    return clusters


# ════════════════════════════════════════════════════════════════════
# Fleiss' kappa
# ════════════════════════════════════════════════════════════════════

def _fleiss_kappa(ratings: List[List[str]], categories: List[str]) -> float:
    """Compute Fleiss' kappa from a list of rating vectors.

    ratings: list of items, each item is a list of raters' category assignments.
    """
    n_items = len(ratings)
    n_raters = len(ratings[0]) if ratings else 0
    k = len(categories)
    if n_items == 0 or n_raters < 2:
        return float("nan")

    cat_idx = {c: i for i, c in enumerate(categories)}

    # Build count matrix (n_items × k)
    counts = np.zeros((n_items, k))
    for i, item_ratings in enumerate(ratings):
        for r in item_ratings:
            if r in cat_idx:
                counts[i, cat_idx[r]] += 1

    # Proportions
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = float(np.sum(p_j ** 2))

    P_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(np.mean(P_i))

    if abs(1 - P_e) < 1e-10:
        return 1.0 if P_bar >= 1.0 - 1e-10 else 0.0
    return (P_bar - P_e) / (1 - P_e)


# ════════════════════════════════════════════════════════════════════
# Main experiment
# ════════════════════════════════════════════════════════════════════

def run_experiment(
    windows: List[Dict],
    ticker: str = "AAPL",
    models: List[Tuple[str, str]] = None,
    api_key: str = "",
    output_dir: str = "results/cross_llm_semantic_test",
    delay: float = 0.5,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    if models is None:
        models = MODELS

    model_names = [m[0] for m in models]
    all_window_results = []
    total_calls = 0
    parse_failures = 0

    # Best example tracking (window with most matched models)
    best_example = None
    best_example_model_count = 0

    for wi, window in enumerate(windows):
        start = window["start_date"]
        end = window["end_date"]
        real_pct = window["price_change_pct"]
        fake_pct = -real_pct

        # Prices
        real_open = 100.0
        real_close = real_open * (1 + real_pct / 100)
        fake_open = real_close
        fake_close = real_open

        # Get RAG news (same for all models / both conditions)
        news_ctx = _get_news_context(ticker, start, end)
        ctx_str = news_ctx if news_ctx else "(No news articles available for this period.)"

        # Build prompts
        real_prompt = _PROMPT.format(
            ticker=ticker, start_date=start, end_date=end,
            open_price=f"{real_open:.2f}", close_price=f"{real_close:.2f}",
            price_change_pct=real_pct, news_context=ctx_str,
            categories=FACTOR_CATEGORIES,
        )
        fake_prompt = _PROMPT.format(
            ticker=ticker, start_date=start, end_date=end,
            open_price=f"{fake_open:.2f}", close_price=f"{fake_close:.2f}",
            price_change_pct=fake_pct, news_context=ctx_str,
            categories=FACTOR_CATEGORIES,
        )

        # ── Call each model with REAL and COUNTERFACTUAL prompts ────
        real_factors_by_model: Dict[str, List[Dict]] = {}
        fake_factors_by_model: Dict[str, List[Dict]] = {}

        for model_label, model_id in models:
            # Real price call
            try:
                raw_real = _call_llm(real_prompt, model_id, api_key)
                f_real = _parse_factors(raw_real)
                if f_real:
                    real_factors_by_model[model_label] = f_real
                else:
                    parse_failures += 1
                total_calls += 1
            except Exception as e:
                print(f"    ERROR {model_label} real: {e}")
                total_calls += 1
                parse_failures += 1

            time.sleep(delay)

            # Counterfactual call
            try:
                raw_fake = _call_llm(fake_prompt, model_id, api_key)
                f_fake = _parse_factors(raw_fake)
                if f_fake:
                    fake_factors_by_model[model_label] = f_fake
                else:
                    parse_failures += 1
                total_calls += 1
            except Exception as e:
                print(f"    ERROR {model_label} fake: {e}")
                total_calls += 1
                parse_failures += 1

            time.sleep(delay)

        # ── Cluster factors across models (real) ────────────────────
        real_clusters = _cluster_factors(real_factors_by_model)
        fake_clusters = _cluster_factors(fake_factors_by_model)

        # ── Match real clusters ↔ fake clusters ─────────────────────
        matched_clusters = []
        used_fake = set()

        for rc in real_clusters:
            best_sim = 0.0
            best_fc_idx = -1
            for fi, fc in enumerate(fake_clusters):
                if fi in used_fake:
                    continue
                sim = _factor_similarity(rc["canonical_name"], fc["canonical_name"])
                if sim > best_sim:
                    best_sim = sim
                    best_fc_idx = fi
            if best_fc_idx >= 0 and best_sim >= 0.15:
                fc = fake_clusters[best_fc_idx]
                used_fake.add(best_fc_idx)

                # Per-model direction comparison
                model_comparisons = []
                for mn in model_names:
                    if mn in rc["per_model"] and mn in fc["per_model"]:
                        d_real = rc["per_model"][mn]["direction"]
                        d_fake = fc["per_model"][mn]["direction"]
                        model_comparisons.append({
                            "model": mn,
                            "direction_real": d_real,
                            "direction_fake": d_fake,
                            "sign_stable": bool(
                                (abs(d_real) < 0.1 and abs(d_fake) < 0.1) or
                                np.sign(d_real) == np.sign(d_fake)
                            ),
                            "delta": abs(d_real - d_fake),
                            "real_factor_detail": rc["per_model"][mn],
                            "fake_factor_detail": fc["per_model"][mn],
                        })

                if model_comparisons:
                    matched_clusters.append({
                        "real_name": rc["canonical_name"],
                        "fake_name": fc["canonical_name"],
                        "category": rc["category"],
                        "similarity": round(best_sim, 3),
                        "n_models_matched": len(model_comparisons),
                        "model_comparisons": model_comparisons,
                    })

                    # Track best example
                    if len(model_comparisons) > best_example_model_count:
                        best_example_model_count = len(model_comparisons)
                        best_example = {
                            "window": f"{start} → {end}",
                            "real_pct": real_pct,
                            "fake_pct": fake_pct,
                            "cluster": matched_clusters[-1],
                        }

        # ── Per-window summary ──────────────────────────────────────
        wresult = {
            "window": f"{start} → {end}",
            "real_pct": real_pct,
            "fake_pct": fake_pct,
            "n_models_responded_real": len(real_factors_by_model),
            "n_models_responded_fake": len(fake_factors_by_model),
            "n_real_clusters": len(real_clusters),
            "n_fake_clusters": len(fake_clusters),
            "n_matched_clusters": len(matched_clusters),
            "matched_clusters": matched_clusters,
            "real_factors_by_model": {
                m: [{"name": f["name"], "direction": f["direction"],
                     "category": f["category"], "description": f["description"]}
                    for f in fs]
                for m, fs in real_factors_by_model.items()
            },
            "fake_factors_by_model": {
                m: [{"name": f["name"], "direction": f["direction"],
                     "category": f["category"], "description": f["description"]}
                    for f in fs]
                for m, fs in fake_factors_by_model.items()
            },
        }
        all_window_results.append(wresult)

        n_total_matched = sum(len(w["matched_clusters"]) for w in all_window_results)
        print(f"  [{wi+1}/{len(windows)}] {start}→{end}  "
              f"pct={real_pct:+.1f}%→{fake_pct:+.1f}%  "
              f"clusters_matched={len(matched_clusters)}  "
              f"total={n_total_matched}")

    # ════════════════════════════════════════════════════════════════
    # Aggregate metrics
    # ════════════════════════════════════════════════════════════════

    # 1. Per-model metrics
    per_model_dirs_real = defaultdict(list)
    per_model_dirs_fake = defaultdict(list)
    per_model_sign_stable = defaultdict(list)

    # 2. Cross-model: for Fleiss' κ, each "item" = (cluster, condition)
    #    raters = models, categories = {bullish, bearish, neutral}
    fleiss_real_ratings = []  # list of [model1_cat, model2_cat, ...]
    fleiss_fake_ratings = []
    fleiss_stability_ratings = []  # per-cluster: each model votes "semantic" or "price_following"

    for wr in all_window_results:
        for mc in wr["matched_clusters"]:
            for comp in mc["model_comparisons"]:
                mn = comp["model"]
                per_model_dirs_real[mn].append(comp["direction_real"])
                per_model_dirs_fake[mn].append(comp["direction_fake"])
                per_model_sign_stable[mn].append(1.0 if comp["sign_stable"] else 0.0)

            # Fleiss' κ: direction classification
            if mc["n_models_matched"] >= 2:
                real_row = []
                fake_row = []
                stability_row = []
                for mn in model_names:
                    match = next((c for c in mc["model_comparisons"] if c["model"] == mn), None)
                    if match:
                        # Classify direction
                        dr = match["direction_real"]
                        df = match["direction_fake"]
                        real_row.append("bullish" if dr > 0.15 else ("bearish" if dr < -0.15 else "neutral"))
                        fake_row.append("bullish" if df > 0.15 else ("bearish" if df < -0.15 else "neutral"))
                        stability_row.append("semantic" if match["sign_stable"] else "price_following")
                if len(real_row) >= 2:
                    fleiss_real_ratings.append(real_row)
                    fleiss_fake_ratings.append(fake_row)
                    fleiss_stability_ratings.append(stability_row)

    # Per-model stats
    from scipy.stats import pearsonr, spearmanr
    per_model_stats = {}
    for mn in model_names:
        dr = per_model_dirs_real.get(mn, [])
        df = per_model_dirs_fake.get(mn, [])
        ss = per_model_sign_stable.get(mn, [])
        if len(dr) >= 3:
            pr, pp = pearsonr(dr, df)
            sr, sp = spearmanr(dr, df)
        else:
            pr = pp = sr = sp = float("nan")
        per_model_stats[mn] = {
            "n_factors": len(dr),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 6) if not np.isnan(pp) else None,
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 6) if not np.isnan(sp) else None,
            "sign_agreement": round(float(np.mean(ss)), 4) if ss else None,
            "mean_delta": round(float(np.mean([abs(r - f) for r, f in zip(dr, df)])), 4) if dr else None,
        }

    # Aggregate across all models
    all_dr = []
    all_df = []
    all_ss = []
    for mn in model_names:
        all_dr.extend(per_model_dirs_real.get(mn, []))
        all_df.extend(per_model_dirs_fake.get(mn, []))
        all_ss.extend(per_model_sign_stable.get(mn, []))

    if len(all_dr) >= 3:
        agg_pr, agg_pp = pearsonr(all_dr, all_df)
        agg_sr, agg_sp = spearmanr(all_dr, all_df)
    else:
        agg_pr = agg_pp = agg_sr = agg_sp = float("nan")

    # Classification
    n_semantic = sum(1 for s in all_ss if s > 0.5)
    n_price_following = len(all_ss) - n_semantic
    semantic_pct = 100 * n_semantic / len(all_ss) if all_ss else 0

    # Fleiss' κ
    dir_cats = ["bullish", "bearish", "neutral"]
    stab_cats = ["semantic", "price_following"]

    # Pad ratings to same length (only include items where all rated)
    def _pad_ratings(ratings, n_raters, pad_val="neutral"):
        padded = []
        for row in ratings:
            if len(row) == n_raters:
                padded.append(row)
            elif len(row) >= 2:
                padded.append(row + [pad_val] * (n_raters - len(row)))
        return padded

    n_models = len(model_names)
    kappa_real = _fleiss_kappa(_pad_ratings(fleiss_real_ratings, n_models), dir_cats)
    kappa_fake = _fleiss_kappa(_pad_ratings(fleiss_fake_ratings, n_models), dir_cats)
    kappa_stability = _fleiss_kappa(
        _pad_ratings(fleiss_stability_ratings, n_models, "semantic"), stab_cats
    )

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "experiment": "Cross-LLM Counterfactual Price Injection — Semantic Direction Test",
        "ticker": ticker,
        "models": [{"label": m[0], "id": m[1]} for m in models],
        "n_windows": len(windows),
        "total_llm_calls": total_calls,
        "total_parse_failures": parse_failures,
        "aggregate_metrics": {
            "pearson_r": round(float(agg_pr), 4) if not np.isnan(agg_pr) else None,
            "pearson_p": round(float(agg_pp), 6) if not np.isnan(agg_pp) else None,
            "spearman_r": round(float(agg_sr), 4) if not np.isnan(agg_sr) else None,
            "spearman_p": round(float(agg_sp), 6) if not np.isnan(agg_sp) else None,
            "sign_agreement": round(float(np.mean(all_ss)), 4) if all_ss else None,
            "mean_delta": round(float(np.mean([abs(r - f) for r, f in zip(all_dr, all_df)])), 4) if all_dr else None,
            "n_semantic": n_semantic,
            "n_price_following": n_price_following,
            "semantic_pct": round(semantic_pct, 1),
            "total_factor_comparisons": len(all_ss),
        },
        "cross_llm_concordance": {
            "fleiss_kappa_real_direction": round(float(kappa_real), 4) if not np.isnan(kappa_real) else None,
            "fleiss_kappa_fake_direction": round(float(kappa_fake), 4) if not np.isnan(kappa_fake) else None,
            "fleiss_kappa_stability_class": round(float(kappa_stability), 4) if not np.isnan(kappa_stability) else None,
            "n_items_rated": len(fleiss_real_ratings),
        },
        "per_model": per_model_stats,
        "best_example": best_example,
        "per_window": all_window_results,
    }

    out_path = os.path.join(output_dir, "cross_llm_semantic_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Print ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-LLM SEMANTIC DIRECTION TEST — RESULTS")
    print("=" * 70)
    print(f"Windows: {len(windows)} | LLM calls: {total_calls} "
          f"({parse_failures} parse failures)")
    print(f"Total factor comparisons: {len(all_ss)}")
    print()
    print("─── Aggregate (all models pooled) ───")
    print(f"  Pearson ρ:        {agg_pr:.4f}  (p={agg_pp:.2e})")
    print(f"  Spearman ρ:       {agg_sr:.4f}  (p={agg_sp:.2e})")
    print(f"  Sign agreement:   {np.mean(all_ss):.1%}")
    print(f"  Semantic:         {n_semantic} ({semantic_pct:.1f}%)")
    print(f"  Price-following:  {n_price_following} ({100-semantic_pct:.1f}%)")
    print()
    print("─── Per-model ───")
    print(f"  {'Model':<20s} {'N':>4s} {'ρ':>7s} {'Sign%':>7s} {'Δ':>7s}")
    for mn in model_names:
        s = per_model_stats[mn]
        print(f"  {mn:<20s} {s['n_factors']:4d} "
              f"{s['pearson_r']:7.3f} "
              f"{(s['sign_agreement'] or 0)*100:6.1f}% "
              f"{s['mean_delta'] or 0:7.3f}")
    print()
    print("─── Cross-LLM Concordance (Fleiss' κ) ───")
    print(f"  Real direction κ:       {kappa_real:.4f}")
    print(f"  Counterfactual dir κ:   {kappa_fake:.4f}")
    print(f"  Stability class κ:      {kappa_stability:.4f}")
    print(f"  (items rated: {len(fleiss_real_ratings)})")
    print()

    # ── Print detailed example ──────────────────────────────────────
    if best_example:
        _print_example(best_example, models)

    print("=" * 70)
    return summary


def _print_example(example: Dict, models: List[Tuple[str, str]]):
    """Print a detailed example showing one window, one factor, all LLMs."""
    print("─── DETAILED EXAMPLE ───")
    print(f"  Window: {example['window']}")
    print(f"  Real price change:  {example['real_pct']:+.2f}%")
    print(f"  Counterfactual:     {example['fake_pct']:+.2f}%")
    cl = example["cluster"]
    print(f"  Factor: {cl['real_name']} ≈ {cl['fake_name']} "
          f"(category: {cl['category']}, sim={cl['similarity']:.2f})")
    print(f"  Models matched: {cl['n_models_matched']}/{len(models)}")
    print()
    print(f"  {'Model':<20s} {'Dir(real)':>10s} {'Dir(fake)':>10s} "
          f"{'Stable?':>8s} {'Δ':>7s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*7}")
    for comp in cl["model_comparisons"]:
        stable = "✓ SEM" if comp["sign_stable"] else "✗ PF"
        print(f"  {comp['model']:<20s} {comp['direction_real']:+10.3f} "
              f"{comp['direction_fake']:+10.3f} {stable:>8s} "
              f"{comp['delta']:7.3f}")
    print()
    # Show each model's description
    print("  Per-model factor descriptions (REAL price):")
    for comp in cl["model_comparisons"]:
        desc = comp.get("real_factor_detail", {}).get("description", "—")
        print(f"    [{comp['model']}] {desc[:120]}")
    print()
    print("  Per-model factor descriptions (COUNTERFACTUAL price):")
    for comp in cl["model_comparisons"]:
        desc = comp.get("fake_factor_detail", {}).get("description", "—")
        print(f"    [{comp['model']}] {desc[:120]}")
    print()


# ════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════

def plot_results(summary: Dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available")
        return

    per_model = summary["per_model"]
    model_names = list(per_model.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle("Cross-LLM Counterfactual Price Injection — Semantic Direction Test",
                 fontsize=14, fontweight="bold")

    # Panel 1: Per-model Pearson ρ bar chart
    ax = axes[0]
    rhos = [per_model[m]["pearson_r"] for m in model_names]
    colors = ["#2ecc71" if r > 0.15 else ("#e74c3c" if r < -0.15 else "#95a5a6") for r in rhos]
    bars = ax.barh(model_names, rhos, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson ρ(real, counterfactual)")
    ax.set_title("Per-Model Correlation")
    ax.set_xlim(-1.1, 1.1)
    ax.grid(True, alpha=0.2, axis="x")
    for bar, v in zip(bars, rhos):
        ax.text(v + 0.02 * np.sign(v), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)

    # Panel 2: Per-model sign agreement
    ax = axes[1]
    sign_rates = [(per_model[m]["sign_agreement"] or 0) * 100 for m in model_names]
    bars = ax.barh(model_names, sign_rates, color="#3498db", edgecolor="white")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xlabel("Sign Agreement (%)")
    ax.set_title("Direction Stability")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")
    for bar, v in zip(bars, sign_rates):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}%", va="center", fontsize=9)

    # Panel 3: Fleiss' κ comparison
    ax = axes[2]
    conc = summary["cross_llm_concordance"]
    kappas = [
        conc.get("fleiss_kappa_real_direction", 0) or 0,
        conc.get("fleiss_kappa_fake_direction", 0) or 0,
        conc.get("fleiss_kappa_stability_class", 0) or 0,
    ]
    labels = ["Real Dir κ", "Counterfactual\nDir κ", "Stability\nClass κ"]
    kcolors = ["#2ecc71", "#e67e22", "#9b59b6"]
    bars = ax.bar(labels, kappas, color=kcolors, edgecolor="white", width=0.6)
    ax.set_ylabel("Fleiss' κ")
    ax.set_title("Cross-LLM Concordance")
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(0.61, color="gray", linestyle=":", alpha=0.5, label="Substantial (0.61)")
    ax.axhline(0.81, color="gray", linestyle="--", alpha=0.5, label="Almost Perfect (0.81)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    for bar, v in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 4: Scatter real vs fake (all models pooled)
    ax = axes[3]
    for wr in summary["per_window"]:
        for mc in wr["matched_clusters"]:
            for comp in mc["model_comparisons"]:
                c = "#2ecc71" if comp["sign_stable"] else "#e74c3c"
                ax.scatter(comp["direction_real"], comp["direction_fake"],
                           c=c, alpha=0.4, s=25, edgecolors="white", linewidths=0.3)
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="d_R = d_F")
    ax.plot([-1, 1], [1, -1], "r:", alpha=0.3, label="d_R = -d_F")
    agg = summary["aggregate_metrics"]
    ax.set_title(f"All Models Pooled\nρ={agg['pearson_r']}, sign={agg['sign_agreement']:.0%}")
    ax.set_xlabel("Direction (real price)")
    ax.set_ylabel("Direction (inverted price)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig_cross_llm_semantic.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures saved to {output_dir}/fig_cross_llm_semantic.{{png,pdf}}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-LLM Counterfactual Price Injection — Semantic Test")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--analyses", default=None)
    parser.add_argument("--n-windows", type=int, default=20,
                        help="Number of windows (default: 20)")
    parser.add_argument("--output-dir", default="results/cross_llm_semantic_test")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.analyses is None:
        analyses_path = str(
            PROJECT_ROOT / "results" / "multi_asset_2018_2023" / args.ticker / "analyses.json"
        )
    else:
        analyses_path = args.analyses

    if not os.path.exists(analyses_path):
        print(f"ERROR: {analyses_path} not found.")
        sys.exit(1)

    print("=" * 70)
    print("CROSS-LLM COUNTERFACTUAL PRICE INJECTION — SEMANTIC DIRECTION TEST")
    print("=" * 70)
    print(f"Ticker:   {args.ticker}")
    print(f"Models:   {len(MODELS)} ({', '.join(m[0] for m in MODELS)})")
    print(f"Windows:  {args.n_windows}")
    print(f"Calls:    ~{args.n_windows * len(MODELS) * 2} "
          f"({args.n_windows} windows × {len(MODELS)} models × 2 conditions)")
    print(f"Output:   {args.output_dir}")
    print()

    windows = select_windows(analyses_path, args.n_windows)
    if not windows:
        print("ERROR: No eligible windows")
        sys.exit(1)

    summary = run_experiment(
        windows=windows,
        ticker=args.ticker,
        models=MODELS,
        output_dir=args.output_dir,
        delay=args.delay,
    )

    plot_results(summary, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
