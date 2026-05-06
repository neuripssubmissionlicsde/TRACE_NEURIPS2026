#!/usr/bin/env python3
"""Tests for semantic concordance script."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_factors():
    """Multiple factors that should cluster semantically."""
    return [
        {"name": "fed_rate_hike",          "category": "macroeconomic",    "direction":  0.8, "magnitude": 0.3, "description": "Federal Reserve raised interest rates by 75bps"},
        {"name": "monetary_tightening",    "category": "macroeconomic",    "direction":  0.7, "magnitude": 0.25, "description": "Federal Reserve monetary policy tightening"},
        {"name": "interest_rate_increase", "category": "macroeconomic",    "direction":  0.9, "magnitude": 0.2, "description": "Interest rates increased significantly"},
        {"name": "oil_price_surge",        "category": "geopolitical",     "direction": -0.6, "magnitude": 0.15, "description": "Oil prices surged due to supply concerns"},
        {"name": "crude_oil_spike",        "category": "geopolitical",     "direction": -0.5, "magnitude": 0.1, "description": "Crude oil prices spiked on geopolitical tensions"},
        {"name": "apple_earnings_beat",    "category": "company_specific", "direction":  0.9, "magnitude": 0.3, "description": "Apple reported earnings above expectations"},
        {"name": "market_noise",           "category": "market_sentiment", "direction":  0.0, "magnitude": 0.1, "description": "Residual variance"},
    ]


@pytest.fixture
def sample_window_factors():
    """Per-window per-model factor data (4 models, 2 windows)."""
    return [
        {
            "GPT-4o-mini": [
                {"name": "fed_rate_hike", "category": "macroeconomic", "direction": 0.8, "magnitude": 0.5},
                {"name": "oil_supply_risk", "category": "geopolitical", "direction": -0.6, "magnitude": 0.3},
            ],
            "Claude-3-Haiku": [
                {"name": "monetary_tightening", "category": "macroeconomic", "direction": 0.7, "magnitude": 0.4},
                {"name": "crude_oil_spike", "category": "geopolitical", "direction": -0.5, "magnitude": 0.3},
            ],
            "Mistral-Small-2603": [
                {"name": "interest_rate_increase", "category": "macroeconomic", "direction": 0.9, "magnitude": 0.6},
            ],
            "Gemini-2.5-Flash": [
                {"name": "fed_rate_hike_75bps", "category": "macroeconomic", "direction": 0.85, "magnitude": 0.5},
                {"name": "oil_price_surge", "category": "geopolitical", "direction": -0.7, "magnitude": 0.3},
            ],
        },
        {
            "GPT-4o-mini": [
                {"name": "earnings_beat", "category": "company_specific", "direction": 0.9, "magnitude": 0.5},
            ],
            "Claude-3-Haiku": [
                {"name": "strong_quarterly_results", "category": "company_specific", "direction": 0.85, "magnitude": 0.5},
            ],
            "Mistral-Small-2603": [
                {"name": "revenue_growth", "category": "company_specific", "direction": 0.8, "magnitude": 0.5},
            ],
            "Gemini-2.5-Flash": [
                {"name": "earnings_beat", "category": "company_specific", "direction": 0.9, "magnitude": 0.5},
            ],
        },
    ]


@pytest.fixture
def mock_embeddings():
    """Fake embeddings that cluster correctly."""
    # 3 rate factors close together, 2 oil factors close, 1 earnings far
    np.random.seed(42)
    return np.array([
        [1.0, 0.0, 0.0] + [0]*381,  # fed_rate_hike
        [0.95, 0.05, 0.0] + [0]*381,  # monetary_tightening
        [0.9, 0.1, 0.0] + [0]*381,   # interest_rate_increase
        [0.0, 1.0, 0.0] + [0]*381,   # oil_price_surge
        [0.05, 0.95, 0.0] + [0]*381, # crude_oil_spike
        [0.0, 0.0, 1.0] + [0]*381,   # apple_earnings_beat
    ], dtype=np.float32)


# ── Tests: Embedding collection ─────────────────────────────────────

class TestCollectUniqueFactor:
    def test_deduplicates_by_name(self, sample_window_factors):
        from run_semantic_concordance import collect_unique_factors
        factors = collect_unique_factors(sample_window_factors)
        names = [f["name"] for f in factors]
        assert len(names) == len(set(names)), "Factor names should be unique"

    def test_excludes_noise(self, sample_window_factors):
        # Add a noise factor
        sample_window_factors[0]["GPT-4o-mini"].append(
            {"name": "market_noise", "category": "market_sentiment",
             "direction": 0.0, "magnitude": 0.1}
        )
        from run_semantic_concordance import collect_unique_factors
        factors = collect_unique_factors(sample_window_factors)
        names = [f["name"] for f in factors]
        assert "market_noise" not in names


# ── Tests: Semantic clustering ───────────────────────────────────────

class TestSemanticClustering:
    def test_produces_clusters(self, sample_factors, mock_embeddings):
        from run_semantic_concordance import cluster_factors_semantic
        non_noise = [f for f in sample_factors if "noise" not in f["name"]]
        clusters = cluster_factors_semantic(non_noise, mock_embeddings, distance_threshold=0.5)
        assert len(clusters) >= 2, "Should produce at least 2 clusters"
        assert len(clusters) <= len(non_noise), "Should not produce more clusters than factors"

    def test_similar_factors_grouped(self, sample_factors, mock_embeddings):
        from run_semantic_concordance import cluster_factors_semantic
        non_noise = [f for f in sample_factors if "noise" not in f["name"]]
        clusters = cluster_factors_semantic(non_noise, mock_embeddings, distance_threshold=0.5)

        # Find which cluster contains fed_rate_hike
        rate_cluster = None
        for c in clusters:
            names = [f["name"] for f in c["factors"]]
            if "fed_rate_hike" in names:
                rate_cluster = c
                break
        assert rate_cluster is not None
        rate_names = [f["name"] for f in rate_cluster["factors"]]
        assert "monetary_tightening" in rate_names or "interest_rate_increase" in rate_names


# ── Tests: Temporal concordance computation ──────────────────────────

class TestTemporalConcordance:
    def test_computes_per_window_agreement(self):
        from run_semantic_concordance import compute_window_concordance
        model_names = ["A", "B", "C"]
        window_factors = {
            "A": [{"name": "fed_rate", "direction": 0.8, "category": "macro", "cluster_id": 0}],
            "B": [{"name": "rate_hike", "direction": 0.7, "category": "macro", "cluster_id": 0}],
            "C": [{"name": "rate_up",   "direction": -0.5, "category": "macro", "cluster_id": 0}],
        }
        result = compute_window_concordance(window_factors, model_names, n_clusters=1)
        assert len(result) == 1  # One cluster
        r = result[0]
        assert r["coverage"] == 1.0  # All 3 models
        # 2/3 agree on bullish, 1 is bearish
        assert 0.5 < r["direction_agreement"] < 1.0

    def test_partial_coverage(self):
        from run_semantic_concordance import compute_window_concordance
        model_names = ["A", "B", "C", "D"]
        window_factors = {
            "A": [{"name": "x", "direction": 0.8, "category": "macro", "cluster_id": 0}],
            "B": [{"name": "y", "direction": 0.7, "category": "macro", "cluster_id": 0}],
        }
        result = compute_window_concordance(window_factors, model_names, n_clusters=1)
        assert result[0]["coverage"] == 0.5  # 2 out of 4 models


# ── Tests: Fleiss kappa temporal ─────────────────────────────────────

class TestFleissKappaTemporal:
    def test_perfect_agreement(self):
        from run_semantic_concordance import fleiss_kappa
        ratings = [
            ["bullish", "bullish", "bullish"],
            ["bearish", "bearish", "bearish"],
        ]
        k = fleiss_kappa(ratings, ["bullish", "bearish", "neutral"])
        assert k > 0.9

    def test_random_agreement(self):
        from run_semantic_concordance import fleiss_kappa
        np.random.seed(42)
        cats = ["bullish", "bearish", "neutral"]
        ratings = [[cats[np.random.randint(3)] for _ in range(5)] for _ in range(50)]
        k = fleiss_kappa(ratings, cats)
        assert -0.2 < k < 0.3  # near zero for random
