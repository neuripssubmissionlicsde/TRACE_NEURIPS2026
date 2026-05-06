#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-LLM Concordance Heatmap — Pipeline Mode Only (Price + RAG).

Runs the EXACT same pipeline extraction prompt (price data + RAG news)
across multiple LLMs and computes pairwise directional concordance.

Models:
  - GPT-4o-mini       (primary, already cached from main run)
  - Claude-3-Haiku
  - Mistral-Small-2603
  - Gemini-2.5-Flash

Output:
  results/final_run_gpt4o_mini/concordance/
    concordance_results.json
    heatmap_concordance.png / .pdf
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
    ("GPT-4o-mini",       "openrouter/openai/gpt-4o-mini"),
    ("Claude-3-Haiku",    "openrouter/anthropic/claude-3-haiku"),
    ("Mistral-Small-2603","openrouter/mistralai/mistral-small-2603"),
    ("Gemini-2.5-Flash",  "openrouter/google/gemini-2.5-flash"),
]

MODEL_SHORT = {
    "GPT-4o-mini":        "GPT-4o-mini",
    "Claude-3-Haiku":     "Claude-3-Haiku",
    "Mistral-Small-2603": "Mistral-Small",
    "Gemini-2.5-Flash":   "Gemini-2.5-Flash",
}

N_SAMPLES = 5  # LLM calls per window, averaged

# ── Prompt (same as pipeline factor_extractor) ──────────────────────
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


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

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
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credits" in err_str.lower():
                raise
            if attempt == max_retries:
                raise
            wait = 5 * (2 ** attempt)
            print(f"      Retry {attempt+1}/{max_retries} after {type(e).__name__} — waiting {wait}s")
            time.sleep(wait)


def _call_llm_n_samples(prompt: str, model: str, api_key: str,
                        n_samples: int = 5, temperature: float = 0.05,
                        delay: float = 0.3) -> List[List[Dict]]:
    """Call LLM n_samples times and return all parsed factor lists."""
    all_results = []
    for _ in range(n_samples):
        try:
            raw = _call_llm(prompt, model, api_key, temperature)
            factors = _parse_factors(raw)
            if factors:
                all_results.append(factors)
        except Exception:
            pass
        time.sleep(delay)
    return all_results


def _average_factors(all_samples: List[List[Dict]]) -> List[Dict]:
    """Average factors across multiple LLM samples (same logic as pipeline)."""
    if not all_samples:
        return []
    if len(all_samples) == 1:
        return all_samples[0]

    # Collect all factor names and average their attributes
    factor_data = defaultdict(lambda: {
        "directions": [], "magnitudes": [], "confidences": [],
        "categories": [], "descriptions": [],
    })
    for sample in all_samples:
        for f in sample:
            name = f["name"]
            factor_data[name]["directions"].append(f.get("direction", 0))
            factor_data[name]["magnitudes"].append(f.get("magnitude", 0))
            factor_data[name]["confidences"].append(f.get("confidence", 0.5))
            factor_data[name]["categories"].append(f.get("category", "unknown"))
            factor_data[name]["descriptions"].append(f.get("description", ""))

    result = []
    for name, data in factor_data.items():
        # Only keep factors that appear in at least half the samples
        if len(data["directions"]) < len(all_samples) / 2:
            continue
        from collections import Counter
        most_common_cat = Counter(data["categories"]).most_common(1)[0][0]
        result.append({
            "name": name,
            "direction": float(np.mean(data["directions"])),
            "magnitude": float(np.mean(data["magnitudes"])),
            "confidence": float(np.mean(data["confidences"])),
            "category": most_common_cat,
            "description": data["descriptions"][0],
        })

    # Re-normalise magnitudes
    total_mag = sum(f["magnitude"] for f in result)
    if total_mag > 0:
        for f in result:
            f["magnitude"] /= total_mag

    return result


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
                "confidence": float(f.get("confidence", 0.5)),
                "description": str(f.get("description", "")),
            })
        return out
    except Exception:
        return []


_STOP = frozenset({
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "is",
    "market", "stock", "price", "impact", "effect", "factor", "data",
    "change", "movement", "related", "general", "overall",
})


def _tokens(name: str) -> set:
    return set(name.lower().replace("-", "_").split("_")) - _STOP


def _factor_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _get_news_context(ticker, start, end, rag_dir="cache/rag_store", max_articles=50):
    # Try ChromaDB vector store first
    try:
        from sde_causal_generator.news_rag import NewsRAGManager
        rag = NewsRAGManager(persist_dir=rag_dir)
        ctx = rag.get_context(ticker=ticker, start_date=start, end_date=end,
                              max_articles=max_articles)
        if ctx and "[No news articles found" not in ctx:
            return ctx
    except Exception:
        pass

    # Fallback: read directly from article cache JSON files
    return _get_news_from_cache_files(ticker, start, end, rag_dir, max_articles)


def _get_news_from_cache_files(ticker, start, end, rag_dir, max_articles):
    """Load articles directly from cached JSON files (bypass ChromaDB)."""
    cache_dir = os.path.join(rag_dir, "article_cache")
    if not os.path.isdir(cache_dir):
        return ""
    articles = []
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".json"):
            continue
        # Match files that contain this ticker
        if f"_{ticker}_" not in fname and f"_{ticker.lower()}_" not in fname:
            continue
        try:
            with open(os.path.join(cache_dir, fname)) as f:
                data = json.load(f)
            if isinstance(data, list):
                articles.extend(data)
        except Exception:
            continue
    if not articles:
        return ""
    # Filter by date range
    filtered = []
    for a in articles:
        pub = a.get("published_date", "")
        if pub and start <= pub <= end:
            filtered.append(a)
    if not filtered:
        return ""
    filtered.sort(key=lambda a: a.get("published_date", ""))
    filtered = filtered[:max_articles]
    parts = []
    for a in filtered:
        title = a.get("title", "")
        content = a.get("content", "")
        date = a.get("published_date", "")
        parts.append(f"[{date}] {title}\n{content}")
    return "\n\n".join(parts)


def _cluster_factors_for_concordance(
    window_factors: Dict[str, List[Dict]],
    model_names: List[str],
    sim_threshold: float = 0.25,
) -> List[Dict[str, Dict]]:
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


def compute_pairwise_concordance(
    all_window_factors: List[Dict[str, List[Dict]]],
    model_names: List[str],
) -> Tuple[np.ndarray, float]:
    n = len(model_names)
    agree_count = np.zeros((n, n))
    total_count = np.zeros((n, n))
    fleiss_ratings = []

    for wf in all_window_factors:
        clusters = _cluster_factors_for_concordance(wf, model_names)
        for cluster in clusters:
            dir_class = {}
            for mn in model_names:
                if mn in cluster:
                    d = cluster[mn]["direction"]
                    if d > 0.15:
                        dir_class[mn] = "bullish"
                    elif d < -0.15:
                        dir_class[mn] = "bearish"
                    else:
                        dir_class[mn] = "neutral"
            for i_idx, mi in enumerate(model_names):
                for j_idx, mj in enumerate(model_names):
                    if mi in dir_class and mj in dir_class:
                        total_count[i_idx, j_idx] += 1
                        if dir_class[mi] == dir_class[mj]:
                            agree_count[i_idx, j_idx] += 1
            present = [mn for mn in model_names if mn in dir_class]
            if len(present) >= 2:
                row = [dir_class.get(mn, "neutral") for mn in model_names]
                fleiss_ratings.append(row)

    concordance = np.where(total_count > 0, agree_count / total_count, 0.0)
    kappa = _fleiss_kappa(fleiss_ratings, ["bullish", "bearish", "neutral"])
    return concordance, kappa


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
# Plotting
# ════════════════════════════════════════════════════════════════════

def plot_heatmap(
    concordance: np.ndarray,
    kappa: float,
    model_labels: List[str],
    title: str,
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n = len(model_labels)
    colors = ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354",
              "#006d2c", "#00441b"]
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(concordance, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

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

    for i in range(n):
        for j in range(n):
            val = concordance[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val*100:.0f}%",
                    ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Agreement Rate")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = output_path.rsplit(".", 1)[0] + f".{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-LLM Concordance Heatmap — Pipeline Mode (Price + RAG)")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--analyses", default=None,
                        help="Path to analyses.json from main run")
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--output-dir",
                        default="results/final_run_gpt4o_mini/concordance")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="LLM calls per window per model (averaged)")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Locate analyses.json
    analyses_path = args.analyses
    if not analyses_path:
        analyses_path = str(
            PROJECT_ROOT / "results" / "final_run_gpt4o_mini"
            / args.ticker / "analyses.json"
        )
    if not os.path.exists(analyses_path):
        print(f"ERROR: {analyses_path} not found. Run main pipeline first.")
        sys.exit(1)

    # Select windows
    windows = select_windows(analyses_path, n_windows=args.n_windows)

    model_names = [m[0] for m in MODELS]
    short_labels = [MODEL_SHORT.get(m, m) for m in model_names]

    print("=" * 70)
    print("CROSS-LLM CONCORDANCE — PIPELINE MODE (PRICE + RAG)")
    print("=" * 70)
    print(f"Ticker:      {args.ticker}")
    print(f"Models:      {len(MODELS)} ({', '.join(model_names)})")
    print(f"Windows:     {len(windows)}")
    print(f"Samples/win: {args.n_samples}")
    print("=" * 70)

    # Checkpoint support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    all_window_factors = []
    all_window_raw = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_window_factors = ckpt.get("window_factors", [])
        all_window_raw = ckpt.get("window_raw", [])
        start_idx = len(all_window_factors)
        print(f"Resuming from checkpoint: {start_idx}/{len(windows)} done")

    t_start = time.time()
    total_calls = 0
    parse_failures = 0

    for wi, window in enumerate(windows):
        if wi < start_idx:
            continue

        start = window["start_date"]
        end = window["end_date"]
        pct = window["price_change_pct"]
        real_open = 100.0
        real_close = real_open * (1 + pct / 100)

        # Get RAG news
        news_ctx = _get_news_context(args.ticker, start, end)
        ctx_str = news_ctx if news_ctx else "(No news articles available for this period.)"

        prompt = _PROMPT.format(
            ticker=args.ticker, start_date=start, end_date=end,
            open_price=f"{real_open:.2f}", close_price=f"{real_close:.2f}",
            price_change_pct=pct, news_context=ctx_str,
            categories=FACTOR_CATEGORIES,
        )

        window_factors: Dict[str, List[Dict]] = {}
        window_raw_data: Dict[str, Dict] = {}

        for model_label, model_id in MODELS:
            samples = _call_llm_n_samples(
                prompt, model_id, api_key,
                n_samples=args.n_samples,
                temperature=0.05,
                delay=args.delay,
            )
            total_calls += args.n_samples
            parse_failures += args.n_samples - len(samples)

            averaged = _average_factors(samples)
            if averaged:
                window_factors[model_label] = averaged
                window_raw_data[model_label] = {
                    "n_samples_ok": len(samples),
                    "n_samples_total": args.n_samples,
                    "averaged_factors": averaged,
                    "all_samples": [[
                        {"name": f["name"], "direction": f["direction"],
                         "magnitude": f["magnitude"], "category": f["category"]}
                        for f in sample
                    ] for sample in samples],
                }

            time.sleep(0.5)

        all_window_factors.append(window_factors)
        all_window_raw.append({
            "window": f"{start} → {end}",
            "price_change_pct": pct,
            "models_responded": list(window_factors.keys()),
            "per_model": window_raw_data,
        })

        print(f"  [{wi+1}/{len(windows)}] {start}→{end} pct={pct:+.1f}% "
              f"models={len(window_factors)}/{len(MODELS)}")

        # Checkpoint
        with open(ckpt_path, "w") as f:
            json.dump({
                "window_factors": all_window_factors,
                "window_raw": all_window_raw,
            }, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── Compute concordance ─────────────────────────────────────────
    concordance, kappa = compute_pairwise_concordance(
        all_window_factors, model_names
    )

    # ── Save results ────────────────────────────────────────────────
    results = {
        "experiment": "Cross-LLM Concordance — Pipeline Mode (Price + RAG)",
        "ticker": args.ticker,
        "models": [{"label": m[0], "id": m[1]} for m in MODELS],
        "n_windows": len(windows),
        "n_samples_per_window": args.n_samples,
        "total_llm_calls": total_calls,
        "total_parse_failures": parse_failures,
        "elapsed_seconds": round(elapsed, 1),
        "fleiss_kappa": round(float(kappa), 4) if not np.isnan(kappa) else None,
        "concordance_matrix": concordance.tolist(),
        "model_labels": model_names,
        "per_window": all_window_raw,
    }

    out_json = os.path.join(args.output_dir, "concordance_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")

    # ── Plot ────────────────────────────────────────────────────────
    plot_heatmap(
        concordance, kappa, short_labels,
        f"Cross-LLM Directional Concordance — {args.ticker}\n"
        f"(Price + RAG, {args.n_samples} samples/window)",
        os.path.join(args.output_dir, "heatmap_concordance.png"),
    )

    # ── Print summary ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("CONCORDANCE RESULTS")
    print("=" * 70)
    print(f"  Fleiss' κ = {kappa:.4f}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Parse failures:  {parse_failures}")
    print(f"  Elapsed:         {elapsed:.0f}s")
    print()
    print("  Pairwise agreement matrix:")
    header = "          " + "  ".join(f"{s:>12s}" for s in short_labels)
    print(header)
    for i, label in enumerate(short_labels):
        row = f"  {label:>8s}" + "  ".join(
            f"{concordance[i,j]*100:11.0f}%" for j in range(len(short_labels))
        )
        print(row)
    print("=" * 70)

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
