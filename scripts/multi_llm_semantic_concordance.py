#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Temporal Concordance — Cross-LLM Factor Agreement Over Time.

Replaces the simple 4×4 directional concordance heatmap with:
  A) Temporal concordance plot — how cross-LLM agreement evolves over time
  B) Semantic factor clustering — group factors by meaning (embeddings),
     not by lexical token overlap

Uses:
  - sentence-transformers (local, all-MiniLM-L6-v2) for factor embeddings
  - OpenRouter via litellm for multi-model factor extraction
  - Hierarchical clustering (cosine distance) for semantic grouping

Output:
  results/final_run_gpt4o_mini/semantic_concordance/
    semantic_concordance_report.json
    temporal_heatmap.{png,pdf}
    factor_dendrogram.{png,pdf}
    rolling_kappa.{png,pdf}
    concordance_heatmap_semantic.{png,pdf}

Usage:
    python scripts/multi_llm_semantic_concordance.py
    python scripts/multi_llm_semantic_concordance.py --ticker AAPL --n-windows 0  # all windows
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

FACTOR_CATEGORIES = [
    "macroeconomic", "geopolitical", "sector_specific",
    "company_specific", "market_sentiment", "technical", "liquidity",
]

MODELS = [
    ("GPT-4o-mini",        "openrouter/openai/gpt-4o-mini"),
    ("Claude-3-Haiku",     "openrouter/anthropic/claude-3-haiku"),
    ("Mistral-Small-2603", "openrouter/mistralai/mistral-small-2603"),
    ("Gemini-2.5-Flash",   "openrouter/google/gemini-2.5-flash"),
]

MODEL_SHORT = {
    "GPT-4o-mini":        "GPT-4o-mini",
    "Claude-3-Haiku":     "Claude-3-Haiku",
    "Mistral-Small-2603": "Mistral-Small",
    "Gemini-2.5-Flash":   "Gemini-2.5-Flash",
}

N_SAMPLES = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
DISTANCE_THRESHOLD = 0.55  # cosine distance threshold for cutting dendrogram

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


# ═══════════════════════════════════════════════════════════════════
# LLM helpers (reused from concordance)
# ═══════════════════════════════════════════════════════════════════

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
                timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credits" in err_str.lower():
                raise
            if attempt == max_retries:
                raise
            wait = 5 * (2 ** attempt)
            print(f"      Retry {attempt+1}/{max_retries} — waiting {wait}s")
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
        return [{
            "name": str(f.get("name", "unknown")),
            "direction": float(f.get("direction", 0)),
            "magnitude": float(f.get("magnitude", 0.5)),
            "category": str(f.get("category", "unknown")),
            "confidence": float(f.get("confidence", 0.5)),
            "description": str(f.get("description", "")),
        } for f in factors]
    except Exception:
        return []


def _call_llm_n_samples(prompt: str, model: str, api_key: str,
                        n_samples: int = 5, delay: float = 0.3) -> List[List[Dict]]:
    results = []
    for _ in range(n_samples):
        try:
            raw = _call_llm(prompt, model, api_key, temperature=0.05)
            factors = _parse_factors(raw)
            if factors:
                results.append(factors)
        except Exception:
            pass
        time.sleep(delay)
    return results


def _average_factors(all_samples: List[List[Dict]]) -> List[Dict]:
    if not all_samples:
        return []
    if len(all_samples) == 1:
        return all_samples[0]

    factor_data = defaultdict(lambda: {
        "directions": [], "magnitudes": [], "confidences": [],
        "categories": [], "descriptions": [],
    })
    for sample in all_samples:
        for f in sample:
            fd = factor_data[f["name"]]
            fd["directions"].append(f.get("direction", 0))
            fd["magnitudes"].append(f.get("magnitude", 0))
            fd["confidences"].append(f.get("confidence", 0.5))
            fd["categories"].append(f.get("category", "unknown"))
            fd["descriptions"].append(f.get("description", ""))

    result = []
    for name, data in factor_data.items():
        if len(data["directions"]) < len(all_samples) / 2:
            continue
        cat = Counter(data["categories"]).most_common(1)[0][0]
        result.append({
            "name": name,
            "direction": float(np.mean(data["directions"])),
            "magnitude": float(np.mean(data["magnitudes"])),
            "confidence": float(np.mean(data["confidences"])),
            "category": cat,
            "description": data["descriptions"][0],
        })

    total_mag = sum(f["magnitude"] for f in result)
    if total_mag > 0:
        for f in result:
            f["magnitude"] /= total_mag
    return result


def _get_news_context(ticker, start, end, rag_dir="cache/rag_store", max_articles=50):
    # Go directly to cache files — avoids ChromaDB/sentence-transformers reload overhead
    return _get_news_from_cache_files(ticker, start, end, rag_dir, max_articles)


def _get_news_from_cache_files(ticker, start, end, rag_dir, max_articles):
    cache_dir = os.path.join(rag_dir, "article_cache")
    if not os.path.isdir(cache_dir):
        return ""
    articles = []
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".json"):
            continue
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
    filtered = [a for a in articles
                if a.get("published_date", "") and start <= a["published_date"] <= end]
    if not filtered:
        return ""
    filtered.sort(key=lambda a: a.get("published_date", ""))
    filtered = filtered[:max_articles]
    return "\n\n".join(
        f"[{a.get('published_date', '')}] {a.get('title', '')}\n{a.get('content', '')}"
        for a in filtered
    )


# ═══════════════════════════════════════════════════════════════════
# Semantic embedding & clustering
# ═══════════════════════════════════════════════════════════════════

def collect_unique_factors(all_window_factors: List[Dict[str, List[Dict]]]) -> List[Dict]:
    """Collect unique non-noise factors across all windows and models."""
    seen = {}
    for wf in all_window_factors:
        for model_name, factors in wf.items():
            for f in factors:
                name = f["name"]
                if "noise" in name or "consolidation" in name:
                    continue
                if name not in seen:
                    seen[name] = {
                        "name": name,
                        "category": f.get("category", "unknown"),
                        "description": f.get("description", ""),
                    }
    return list(seen.values())


def get_embeddings(texts: List[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Get embeddings using sentence-transformers (local model)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def _factor_embed_text(f: Dict) -> str:
    """Build text for embedding: human-readable name + description."""
    human_name = f["name"].replace("_", " ")
    cat = f.get("category", "")
    desc = f.get("description", "")
    # Remove confidence prefix from description
    desc = re.sub(r"^\[conf=[0-9.]+\]\s*", "", desc)
    return f"{human_name} ({cat}): {desc}" if desc else f"{human_name} ({cat})"


def cluster_factors_semantic(
    factors: List[Dict],
    embeddings: np.ndarray,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> List[Dict]:
    """Hierarchical clustering of factors based on embedding cosine distance."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(factors) <= 1:
        return [{"id": 0, "label": factors[0]["name"] if factors else "",
                 "factors": factors}]

    # Cosine distance
    dists = pdist(embeddings, metric="cosine")
    dists = np.clip(dists, 0, 2)  # numerical safety

    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    clusters_dict: Dict[int, List[Dict]] = defaultdict(list)
    for i, lab in enumerate(labels):
        factors[i]["_cluster_id"] = int(lab) - 1
        clusters_dict[int(lab) - 1].append(factors[i])

    clusters = []
    for cid in sorted(clusters_dict.keys()):
        members = clusters_dict[cid]
        cats = Counter(f.get("category", "") for f in members)
        top_cat = cats.most_common(1)[0][0]
        # Representative = shortest descriptive name (heuristic: most general)
        names = sorted([f["name"] for f in members], key=len)
        # Prefer a name with >1 token (skip single-word like "technical")
        rep = names[0]
        for n in names:
            if "_" in n:
                rep = n
                break
        label = f"{rep.replace('_', ' ').title()} [{top_cat}]"
        clusters.append({
            "id": cid,
            "label": label,
            "category": top_cat,
            "factors": members,
            "factor_names": [f["name"] for f in members],
            "size": len(members),
        })

    return clusters


def build_linkage_data(factors: List[Dict], embeddings: np.ndarray):
    """Build linkage matrix for dendrogram plotting."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    dists = pdist(embeddings, metric="cosine")
    dists = np.clip(dists, 0, 2)
    Z = linkage(dists, method="average")
    return Z


# ═══════════════════════════════════════════════════════════════════
# Concordance computation
# ═══════════════════════════════════════════════════════════════════

def assign_factors_to_clusters(
    window_factors: Dict[str, List[Dict]],
    factor_cluster_map: Dict[str, int],
    embeddings_cache: Dict[str, np.ndarray],
    cluster_centroids: Dict[int, np.ndarray],
) -> Dict[str, List[Dict]]:
    """For each model's factors, assign them to semantic clusters.

    Factors whose name matches exactly go to their known cluster.
    New/unknown factors get assigned to the nearest cluster by embedding similarity.
    """
    result = {}
    for model_name, factors in window_factors.items():
        assigned = []
        for f in factors:
            if "noise" in f["name"] or "consolidation" in f["name"]:
                continue
            if f["name"] in factor_cluster_map:
                f_copy = dict(f)
                f_copy["cluster_id"] = factor_cluster_map[f["name"]]
                assigned.append(f_copy)
            elif embeddings_cache and cluster_centroids:
                # Compute embedding for this unseen factor
                text = _factor_embed_text(f)
                try:
                    emb = get_embeddings([text])[0]
                    best_cid = min(cluster_centroids.keys(),
                                   key=lambda c: float(np.dot(emb, cluster_centroids[c])))
                    f_copy = dict(f)
                    f_copy["cluster_id"] = best_cid
                    assigned.append(f_copy)
                except Exception:
                    pass
        result[model_name] = assigned
    return result


def compute_window_concordance(
    window_factors: Dict[str, List[Dict]],
    model_names: List[str],
    n_clusters: int,
) -> List[Dict]:
    """Compute per-cluster concordance for a single window."""
    results = []
    for cid in range(n_clusters):
        models_with = {}
        for mn in model_names:
            factors = window_factors.get(mn, [])
            cluster_factors = [f for f in factors if f.get("cluster_id") == cid]
            if cluster_factors:
                # Take average direction across matched factors
                avg_dir = float(np.mean([f["direction"] for f in cluster_factors]))
                models_with[mn] = avg_dir

        coverage = len(models_with) / len(model_names) if model_names else 0

        if len(models_with) >= 2:
            directions = list(models_with.values())
            dir_classes = []
            for d in directions:
                if d > 0.15:
                    dir_classes.append("bullish")
                elif d < -0.15:
                    dir_classes.append("bearish")
                else:
                    dir_classes.append("neutral")
            majority = Counter(dir_classes).most_common(1)[0][1]
            direction_agreement = majority / len(dir_classes)
        elif len(models_with) == 1:
            direction_agreement = 1.0
        else:
            direction_agreement = 0.0

        results.append({
            "cluster_id": cid,
            "coverage": coverage,
            "direction_agreement": direction_agreement,
            "n_models": len(models_with),
            "model_directions": models_with,
        })

    return results


def fleiss_kappa(ratings: List[List[str]], categories: List[str]) -> float:
    """Compute Fleiss' kappa for inter-rater agreement."""
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


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_temporal_heatmap(
    temporal_data: List[Dict],
    cluster_labels: List[str],
    window_labels: List[str],
    output_dir: str,
    ticker: str,
):
    """Main temporal concordance heatmap: X=time, Y=semantic cluster, color=score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_clusters = len(cluster_labels)
    n_windows = len(window_labels)

    # Build matrix: rows=clusters, cols=windows
    # Score = coverage × direction_agreement (0=absent/disagree, 1=all agree)
    matrix = np.full((n_clusters, n_windows), np.nan)

    for wi, wd in enumerate(temporal_data):
        for cr in wd.get("cluster_concordance", []):
            cid = cr["cluster_id"]
            if cid < n_clusters:
                score = cr["coverage"] * cr["direction_agreement"]
                matrix[cid, wi] = score if cr["coverage"] > 0 else np.nan

    # Only keep clusters that appear in at least 3 windows
    active_mask = np.sum(~np.isnan(matrix), axis=1) >= 3
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        print("  ⚠ No clusters with sufficient temporal coverage — skipping heatmap")
        return

    matrix_filtered = matrix[active_indices, :]
    labels_filtered = [cluster_labels[i] for i in active_indices]

    # Sort by first appearance
    first_appear = []
    for row in matrix_filtered:
        nans = np.where(~np.isnan(row))[0]
        first_appear.append(nans[0] if len(nans) > 0 else len(row))
    sort_idx = np.argsort(first_appear)
    matrix_filtered = matrix_filtered[sort_idx]
    labels_filtered = [labels_filtered[i] for i in sort_idx]

    # Truncate long labels
    labels_display = [l[:45] + "…" if len(l) > 45 else l for l in labels_filtered]

    cmap = LinearSegmentedColormap.from_list(
        "concordance",
        ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )

    fig_h = max(8, len(labels_display) * 0.35 + 3)
    fig_w = max(14, n_windows * 0.18 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix_filtered, cmap=cmap, aspect="auto",
                   vmin=0, vmax=1, interpolation="nearest")

    # X-axis: window dates (show every Nth)
    step = max(1, n_windows // 20)
    xtick_pos = list(range(0, n_windows, step))
    xtick_labels = [window_labels[i] for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=60, ha="right", fontsize=8)

    ax.set_yticks(range(len(labels_display)))
    ax.set_yticklabels(labels_display, fontsize=8)

    ax.set_xlabel("Time Window", fontsize=11)
    ax.set_ylabel("Semantic Factor Cluster", fontsize=11)
    ax.set_title(
        f"Cross-LLM Temporal Concordance — {ticker}\n"
        f"(Coverage × Direction Agreement, {len(MODELS)} models, {N_SAMPLES} samples/window)",
        fontsize=13, fontweight="bold", pad=15,
    )

    plt.colorbar(im, ax=ax, shrink=0.6, label="Concordance Score", pad=0.02)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"temporal_heatmap.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: temporal_heatmap.png")


def plot_dendrogram(
    Z: np.ndarray,
    factor_labels: List[str],
    cluster_assignments: List[int],
    output_dir: str,
    distance_threshold: float = DISTANCE_THRESHOLD,
):
    """Dendrogram of semantic factor clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig_h = max(10, len(factor_labels) * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Truncate labels for readability
    display_labels = [l.replace("_", " ")[:40] for l in factor_labels]

    dendrogram(
        Z,
        orientation="right",
        labels=display_labels,
        leaf_font_size=7,
        color_threshold=distance_threshold,
        above_threshold_color="#999999",
        ax=ax,
    )

    ax.axvline(x=distance_threshold, color="red", linestyle="--", alpha=0.5,
               label=f"Cut threshold = {distance_threshold:.2f}")
    ax.set_xlabel("Cosine Distance", fontsize=11)
    ax.set_title("Semantic Factor Clustering\n(sentence-transformers embeddings)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"factor_dendrogram.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: factor_dendrogram.png")


def plot_rolling_kappa(
    kappa_values: List[float],
    window_labels: List[str],
    output_dir: str,
    ticker: str,
    roll_size: int = 10,
):
    """Rolling Fleiss' kappa over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(kappa_values) < roll_size:
        print("  ⚠ Not enough windows for rolling kappa plot")
        return

    # Rolling average
    kappa_arr = np.array(kappa_values)
    rolling = np.convolve(kappa_arr, np.ones(roll_size) / roll_size, mode="valid")
    x_roll = np.arange(roll_size - 1, len(kappa_values))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(range(len(kappa_values)), kappa_values, alpha=0.3, s=15,
               color="#2196F3", label="Per-window κ")
    ax.plot(x_roll, rolling, color="#D32F2F", linewidth=2.5,
            label=f"Rolling avg (n={roll_size})")

    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=0.21, color="#FFB300", linestyle="--", alpha=0.4, label="Fair (0.21)")
    ax.axhline(y=0.41, color="#43A047", linestyle="--", alpha=0.4, label="Moderate (0.41)")

    step = max(1, len(window_labels) // 15)
    xtick_pos = list(range(0, len(window_labels), step))
    xtick_lab = [window_labels[i] for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Time Window", fontsize=11)
    ax.set_ylabel("Fleiss' κ (semantic matching)", fontsize=11)
    ax.set_title(f"Rolling Cross-LLM Agreement — {ticker}\n"
                 f"(Fleiss' κ with semantic factor clustering)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"rolling_kappa.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: rolling_kappa.png")


def plot_concordance_heatmap_semantic(
    concordance: np.ndarray,
    kappa: float,
    model_labels: List[str],
    output_dir: str,
    ticker: str,
):
    """4×4 concordance heatmap using semantic matching (for comparison)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n = len(model_labels)
    colors = ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c", "#00441b"]
    cmap = LinearSegmentedColormap.from_list("green", colors, N=256)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(concordance, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    if kappa >= 0.81:
        kl = "Almost perfect"
    elif kappa >= 0.61:
        kl = "Substantial"
    elif kappa >= 0.41:
        kl = "Moderate"
    elif kappa >= 0.21:
        kl = "Fair"
    else:
        kl = "Poor"

    ax.set_title(
        f"Cross-LLM Concordance (Semantic Matching) — {ticker}\n"
        f"(Fleiss' κ = {kappa:.3f} — {kl})",
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
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Agreement Rate")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"concordance_heatmap_semantic.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: concordance_heatmap_semantic.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Temporal Concordance — Cross-LLM Factor Agreement Over Time")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--analyses", default=None)
    parser.add_argument("--n-windows", type=int, default=0,
                        help="0 = use all windows")
    parser.add_argument("--output-dir",
                        default="results/final_run_gpt4o_mini/semantic_concordance")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--distance-threshold", type=float, default=DISTANCE_THRESHOLD)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # ── Load analyses ───────────────────────────────────────────
    analyses_path = args.analyses or str(
        PROJECT_ROOT / "results" / "final_run_gpt4o_mini" / args.ticker / "analyses.json"
    )
    if not os.path.exists(analyses_path):
        print(f"ERROR: {analyses_path} not found")
        sys.exit(1)

    with open(analyses_path) as f:
        all_analyses = json.load(f)

    # Sort chronologically
    all_analyses.sort(key=lambda a: a["start_date"])

    if args.n_windows > 0:
        all_analyses = all_analyses[:args.n_windows]

    n_windows = len(all_analyses)
    model_names = [m[0] for m in MODELS]
    short_labels = [MODEL_SHORT.get(m, m) for m in model_names]

    print("=" * 70)
    print("SEMANTIC TEMPORAL CONCORDANCE")
    print("=" * 70)
    print(f"  Ticker:     {args.ticker}")
    print(f"  Windows:    {n_windows}")
    print(f"  Models:     {len(MODELS)} ({', '.join(model_names)})")
    print(f"  Samples:    {args.n_samples}/window/model")
    print(f"  Embed:      {EMBED_MODEL} (local)")
    print(f"  Dist thresh:{args.distance_threshold}")
    print(f"  Est. calls: {n_windows * (len(MODELS) - 1) * args.n_samples} "
          f"(GPT-4o-mini cached)")
    print("=" * 70)

    # ── Phase 1: Collect multi-model factors per window ─────────
    print("\n── Phase 1: Multi-model factor extraction ──")

    ckpt_path = os.path.join(args.output_dir, "checkpoint_phase1.json")
    all_window_factors: List[Dict[str, List[Dict]]] = []
    all_window_meta: List[Dict] = []
    start_idx = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_window_factors = ckpt.get("window_factors", [])
        all_window_meta = ckpt.get("window_meta", [])
        start_idx = len(all_window_factors)
        print(f"  Resuming from checkpoint: {start_idx}/{n_windows}")

    t_start = time.time()
    total_calls = 0
    parse_failures = 0

    for wi in range(start_idx, n_windows):
        analysis = all_analyses[wi]
        start = analysis["start_date"]
        end = analysis["end_date"]
        pct = analysis["price_change_pct"]

        # GPT-4o-mini: reuse from pipeline (already averaged)
        gpt_factors = [
            {
                "name": f["name"],
                "direction": f["direction"],
                "magnitude": f["magnitude"],
                "category": f["category"],
                "confidence": f.get("confidence", 0.5),
                "description": f.get("description", ""),
            }
            for f in analysis["factors"]
        ]

        window_factors: Dict[str, List[Dict]] = {"GPT-4o-mini": gpt_factors}

        # Other models: call API
        real_open = 100.0
        real_close = real_open * (1 + pct / 100)

        news_ctx = _get_news_context(args.ticker, start, end)
        ctx_str = news_ctx if news_ctx else "(No news articles available for this period.)"

        prompt = _PROMPT.format(
            ticker=args.ticker, start_date=start, end_date=end,
            open_price=f"{real_open:.2f}", close_price=f"{real_close:.2f}",
            price_change_pct=pct, news_context=ctx_str,
            categories=FACTOR_CATEGORIES,
        )

        for model_label, model_id in MODELS:
            if model_label == "GPT-4o-mini":
                continue  # already have it

            samples = _call_llm_n_samples(
                prompt, model_id, api_key,
                n_samples=args.n_samples, delay=args.delay,
            )
            total_calls += args.n_samples
            parse_failures += args.n_samples - len(samples)

            averaged = _average_factors(samples)
            if averaged:
                window_factors[model_label] = averaged
            time.sleep(0.3)

        all_window_factors.append(window_factors)
        all_window_meta.append({
            "window": f"{start} → {end}",
            "start_date": start,
            "end_date": end,
            "price_change_pct": pct,
            "models_responded": list(window_factors.keys()),
        })

        tag = "bull" if pct > 0 else "bear"
        print(f"  [{wi+1}/{n_windows}] {start}→{end} {tag} pct={pct:+.1f}% "
              f"models={len(window_factors)}/{len(MODELS)}")

        # Checkpoint every 5 windows
        if (wi + 1) % 5 == 0 or wi == n_windows - 1:
            with open(ckpt_path, "w") as f:
                json.dump({
                    "window_factors": all_window_factors,
                    "window_meta": all_window_meta,
                }, f, default=str)

    phase1_time = time.time() - t_start
    print(f"\n  Phase 1 done: {total_calls} LLM calls, "
          f"{parse_failures} parse failures, {phase1_time:.0f}s")

    # ── Phase 2: Semantic embedding & clustering ────────────────
    print("\n── Phase 2: Semantic embedding & clustering ──")

    unique_factors = collect_unique_factors(all_window_factors)
    print(f"  Unique non-noise factors: {len(unique_factors)}")

    if len(unique_factors) < 2:
        print("  ⚠ Not enough factors for clustering")
        return

    # Compute embeddings
    texts = [_factor_embed_text(f) for f in unique_factors]
    print(f"  Computing embeddings with {EMBED_MODEL}...")
    embeddings = get_embeddings(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Cluster
    clusters = cluster_factors_semantic(
        unique_factors, embeddings,
        distance_threshold=args.distance_threshold,
    )
    print(f"  Semantic clusters: {len(clusters)}")
    for c in sorted(clusters, key=lambda x: -x["size"])[:15]:
        print(f"    [{c['id']:2d}] {c['label'][:50]:50s} ({c['size']} factors)")

    # Build factor → cluster_id map
    factor_cluster_map = {}
    for c in clusters:
        for fn in c["factor_names"]:
            factor_cluster_map[fn] = c["id"]

    # Cluster centroids for assigning unseen factors
    cluster_centroids = {}
    for c in clusters:
        indices = [i for i, f in enumerate(unique_factors) if f["name"] in c["factor_names"]]
        if indices:
            cluster_centroids[c["id"]] = embeddings[indices].mean(axis=0)

    # Build linkage for dendrogram
    Z = build_linkage_data(unique_factors, embeddings)

    # ── Phase 3: Temporal concordance ───────────────────────────
    print("\n── Phase 3: Temporal concordance ──")

    n_clusters = len(clusters)
    temporal_data = []
    per_window_kappas = []
    window_labels = []

    # For revised 4×4 pairwise concordance (semantic version)
    agree_count = np.zeros((len(model_names), len(model_names)))
    total_count = np.zeros((len(model_names), len(model_names)))
    fleiss_ratings_all = []

    for wi, wf in enumerate(all_window_factors):
        meta = all_window_meta[wi]
        window_labels.append(meta["start_date"][:7])  # YYYY-MM

        # Assign each model's factors to semantic clusters
        assigned = {}
        for mn, factors in wf.items():
            aflist = []
            for f in factors:
                if "noise" in f["name"] or "consolidation" in f["name"]:
                    continue
                f_copy = dict(f)
                f_copy["cluster_id"] = factor_cluster_map.get(f["name"], -1)
                aflist.append(f_copy)
            assigned[mn] = aflist

        # Per-cluster concordance for this window
        cluster_conc = compute_window_concordance(assigned, model_names, n_clusters)

        temporal_data.append({
            "window": meta["window"],
            "start_date": meta["start_date"],
            "price_change_pct": meta["price_change_pct"],
            "cluster_concordance": cluster_conc,
        })

        # Pairwise agreement (semantic version) and Fleiss data
        window_ratings = []
        for cr in cluster_conc:
            if cr["n_models"] < 2:
                continue
            dirs = cr["model_directions"]
            dir_classes = {}
            for mn in model_names:
                if mn in dirs:
                    d = dirs[mn]
                    if d > 0.15:
                        dir_classes[mn] = "bullish"
                    elif d < -0.15:
                        dir_classes[mn] = "bearish"
                    else:
                        dir_classes[mn] = "neutral"

            for i_idx, mi in enumerate(model_names):
                for j_idx, mj in enumerate(model_names):
                    if mi in dir_classes and mj in dir_classes:
                        total_count[i_idx, j_idx] += 1
                        if dir_classes[mi] == dir_classes[mj]:
                            agree_count[i_idx, j_idx] += 1

            present = [mn for mn in model_names if mn in dir_classes]
            if len(present) >= 2:
                row = [dir_classes.get(mn, "neutral") for mn in model_names]
                fleiss_ratings_all.append(row)
                window_ratings.append(row)

        # Per-window kappa
        if len(window_ratings) >= 2:
            wk = fleiss_kappa(window_ratings, ["bullish", "bearish", "neutral"])
            per_window_kappas.append(wk if not np.isnan(wk) else 0.0)
        else:
            per_window_kappas.append(0.0)

    concordance_semantic = np.where(total_count > 0,
                                    agree_count / total_count, 0.0)
    overall_kappa = fleiss_kappa(fleiss_ratings_all, ["bullish", "bearish", "neutral"])
    overall_kappa = 0.0 if np.isnan(overall_kappa) else overall_kappa

    print(f"  Overall Fleiss' κ (semantic): {overall_kappa:.4f}")
    print(f"  Mean per-window κ: {np.mean(per_window_kappas):.4f}")

    # ── Phase 4: Visualizations ─────────────────────────────────
    print("\n── Phase 4: Visualizations ──")

    cluster_labels = [""] * n_clusters
    for c in clusters:
        cluster_labels[c["id"]] = c["label"]

    plot_temporal_heatmap(temporal_data, cluster_labels, window_labels,
                         args.output_dir, args.ticker)

    factor_labels = [f["name"] for f in unique_factors]
    cluster_assignments = [factor_cluster_map.get(f["name"], -1) for f in unique_factors]
    plot_dendrogram(Z, factor_labels, cluster_assignments,
                    args.output_dir, args.distance_threshold)

    plot_rolling_kappa(per_window_kappas, window_labels,
                       args.output_dir, args.ticker)

    plot_concordance_heatmap_semantic(
        concordance_semantic, overall_kappa, short_labels,
        args.output_dir, args.ticker,
    )

    # ── Save report ─────────────────────────────────────────────
    elapsed = time.time() - t_start

    report = {
        "experiment": "Semantic Temporal Concordance",
        "ticker": args.ticker,
        "models": [{"label": m[0], "id": m[1]} for m in MODELS],
        "n_windows": n_windows,
        "n_samples_per_window": args.n_samples,
        "embedding_model": EMBED_MODEL,
        "distance_threshold": args.distance_threshold,
        "total_llm_calls": total_calls,
        "total_parse_failures": parse_failures,
        "elapsed_seconds": round(elapsed, 1),
        "n_unique_factors": len(unique_factors),
        "n_semantic_clusters": len(clusters),
        "overall_fleiss_kappa_semantic": round(overall_kappa, 4),
        "overall_fleiss_kappa_lexical": None,  # will compare with original
        "mean_per_window_kappa": round(float(np.mean(per_window_kappas)), 4),
        "concordance_matrix_semantic": concordance_semantic.tolist(),
        "model_labels": model_names,
        "semantic_clusters": [
            {
                "id": c["id"],
                "label": c["label"],
                "category": c["category"],
                "factor_names": c["factor_names"],
                "size": c["size"],
            }
            for c in sorted(clusters, key=lambda x: -x["size"])
        ],
        "temporal_summary": [
            {
                "window": td["window"],
                "start_date": td["start_date"],
                "price_change_pct": td["price_change_pct"],
                "kappa": round(per_window_kappas[i], 4),
                "active_clusters": sum(
                    1 for cr in td["cluster_concordance"] if cr["coverage"] > 0
                ),
                "mean_concordance": round(float(np.mean([
                    cr["coverage"] * cr["direction_agreement"]
                    for cr in td["cluster_concordance"] if cr["coverage"] > 0
                ])), 4) if any(
                    cr["coverage"] > 0 for cr in td["cluster_concordance"]
                ) else 0.0,
            }
            for i, td in enumerate(temporal_data)
        ],
    }

    report_path = os.path.join(args.output_dir, "semantic_concordance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SEMANTIC CONCORDANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Unique factors:         {len(unique_factors)}")
    print(f"  Semantic clusters:      {len(clusters)}")
    print(f"  Overall κ (semantic):   {overall_kappa:.4f}")
    print(f"  Mean per-window κ:      {np.mean(per_window_kappas):.4f}")
    print(f"  LLM calls:             {total_calls}")
    print(f"  Parse failures:        {parse_failures}")
    print(f"  Elapsed:               {elapsed:.0f}s")
    print(f"  Output:                {args.output_dir}")

    print(f"\n  Pairwise concordance (semantic matching):")
    header = "          " + "  ".join(f"{s:>14s}" for s in short_labels)
    print(header)
    for i, label in enumerate(short_labels):
        row = f"  {label:>8s}" + "  ".join(
            f"{concordance_semantic[i,j]*100:13.0f}%"
            for j in range(len(short_labels))
        )
        print(row)

    print(f"\n  Top semantic clusters:")
    for c in sorted(clusters, key=lambda x: -x["size"])[:10]:
        print(f"    [{c['id']:2d}] {c['label'][:55]} "
              f"({c['size']} factors: {', '.join(c['factor_names'][:3])}{'...' if c['size']>3 else ''})")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
