# -*- coding: utf-8 -*-
"""
Tests for the TimeGAN Baseline Adapter.

All tests use small dummy data and minimal epochs so they run in seconds
without GPU or API keys.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ── Check TimeGAN availability ──────────────────────────────────────
_timegan_available = False
try:
    from sde_causal_generator.timegan_baseline import (
        TimeGANBaseline,
        TimeGANBaselineConfig,
        _timegan_available as _tg_flag,
    )
    _timegan_available = _tg_flag
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _timegan_available,
    reason="TimeGAN implementation not found (emergent-finrl not in workspace)",
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def dummy_ohlcv_multi() -> pd.DataFrame:
    """Multi-ticker OHLCV DataFrame with 150 rows per ticker."""
    rng = np.random.RandomState(42)
    rows = []
    tickers = ["AAA", "BBB"]
    dates = pd.bdate_range("2023-01-02", periods=150, freq="B")

    for tic in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 150)))
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "tic": tic,
                "open": close[i] * (1 + rng.normal(0, 0.003)),
                "high": close[i] * (1 + abs(rng.normal(0, 0.005))),
                "low": close[i] * (1 - abs(rng.normal(0, 0.005))),
                "close": close[i],
                "volume": int(rng.lognormal(15, 0.5)),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def fast_config() -> TimeGANBaselineConfig:
    """Minimal config for fast testing (tiny epochs)."""
    return TimeGANBaselineConfig(
        seq_len=10,
        hidden_dim=8,
        num_layers=1,
        embedding_epochs=2,
        supervised_epochs=2,
        joint_epochs=2,
        batch_size=16,
        device="cpu",
        print_every=1,
    )


# ══════════════════════════════════════════════════════════════════════
# Tests: Configuration
# ══════════════════════════════════════════════════════════════════════


class TestTimeGANBaselineConfig:
    """Tests for TimeGANBaselineConfig."""

    def test_default_config(self):
        cfg = TimeGANBaselineConfig()
        assert cfg.seq_len == 24
        assert cfg.hidden_dim == 24
        assert "close" in cfg.features

    def test_to_timegan_config(self, fast_config):
        tg_cfg = fast_config.to_timegan_config()
        assert tg_cfg.hidden_dim == 8
        assert tg_cfg.embedding_epochs == 2
        assert tg_cfg.num_layers == 1

    def test_custom_features(self):
        cfg = TimeGANBaselineConfig(features=["close", "volume"])
        assert len(cfg.features) == 2


# ══════════════════════════════════════════════════════════════════════
# Tests: Training
# ══════════════════════════════════════════════════════════════════════


class TestTimeGANBaselineTraining:
    """Tests for TimeGAN training."""

    def test_train_single_ticker(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        histories = baseline.train(dummy_ohlcv_multi, tickers=["AAA"])

        assert "AAA" in histories
        assert "AAA" in baseline.trained_tickers
        assert "BBB" not in baseline.trained_tickers

    def test_train_multiple_tickers(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        histories = baseline.train(dummy_ohlcv_multi, tickers=["AAA", "BBB"])

        assert len(baseline.trained_tickers) == 2
        assert "AAA" in histories
        assert "BBB" in histories

    def test_train_all_tickers(self, dummy_ohlcv_multi, fast_config):
        """When tickers=None, train on all unique tickers."""
        baseline = TimeGANBaseline(fast_config)
        histories = baseline.train(dummy_ohlcv_multi)

        assert len(baseline.trained_tickers) == 2

    def test_train_returns_history(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        histories = baseline.train(dummy_ohlcv_multi, tickers=["AAA"])

        hist = histories["AAA"]
        assert "embedding" in hist or "supervised" in hist or "joint" in hist

    def test_train_skips_insufficient_data(self, fast_config):
        """Tickers with fewer rows than seq_len + 10 should be skipped."""
        tiny_df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=5),
            "tic": "TINY",
            "open": [1, 2, 3, 4, 5],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "close": [1, 2, 3, 4, 5],
            "volume": [100] * 5,
        })
        baseline = TimeGANBaseline(fast_config)
        histories = baseline.train(tiny_df, tickers=["TINY"])

        assert len(baseline.trained_tickers) == 0


# ══════════════════════════════════════════════════════════════════════
# Tests: Generation
# ══════════════════════════════════════════════════════════════════════


class TestTimeGANBaselineGeneration:
    """Tests for synthetic data generation."""

    def test_generate_single_ticker(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA"])

        samples = baseline.generate("AAA", n_samples=2, num_trading_days=50)

        assert len(samples) == 2
        for df in samples:
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert "close" in df.columns
            assert "date" in df.columns
            # Prices should be positive
            assert (df["close"] > 0).all()

    def test_generate_all(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA", "BBB"])

        results = baseline.generate_all(n_samples=2, num_trading_days=50)

        assert len(results) == 2
        assert "AAA" in results
        assert "BBB" in results
        for ticker, dfs in results.items():
            assert len(dfs) == 2

    def test_generate_untrained_ticker_raises(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA"])

        with pytest.raises(ValueError, match="not trained"):
            baseline.generate("ZZZ", n_samples=1)

    def test_generated_data_format(self, dummy_ohlcv_multi, fast_config):
        """Output should match the Causal SDE format: date, tic, OHLCV columns."""
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA"])
        samples = baseline.generate("AAA", n_samples=1, num_trading_days=50)

        df = samples[0]
        assert "date" in df.columns
        assert "tic" in df.columns
        assert (df["tic"] == "AAA").all()


# ══════════════════════════════════════════════════════════════════════
# Tests: I/O
# ══════════════════════════════════════════════════════════════════════


class TestTimeGANBaselineIO:
    """Tests for saving and loading."""

    def test_save_results(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA"])
        baseline.generate("AAA", n_samples=2, num_trading_days=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = baseline.save_results(tmpdir)

            assert "AAA" in paths
            assert os.path.exists(os.path.join(tmpdir, "AAA", "synthetic_data_0.csv"))
            assert os.path.exists(os.path.join(tmpdir, "AAA", "synthetic_data_1.csv"))
            assert os.path.exists(os.path.join(tmpdir, "AAA", "synthetic_data_timegan.csv"))
            assert os.path.exists(os.path.join(tmpdir, "AAA", "timegan_report.json"))

            # Verify report JSON is valid
            with open(os.path.join(tmpdir, "AAA", "timegan_report.json")) as f:
                report = json.load(f)
            assert report["ticker"] == "AAA"
            assert report["method"] == "timegan"
            assert report["n_samples"] == 2

    def test_save_and_load_checkpoint(self, dummy_ohlcv_multi, fast_config):
        baseline = TimeGANBaseline(fast_config)
        baseline.train(dummy_ohlcv_multi, tickers=["AAA"])

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline.save_checkpoint(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "timegan_AAA.pt"))

            # Load into new instance
            baseline2 = TimeGANBaseline(fast_config)
            baseline2.load_checkpoint(tmpdir, tickers=["AAA"])
            assert "AAA" in baseline2.trained_tickers

            # Generate from loaded model
            samples = baseline2.generate("AAA", n_samples=1, num_trading_days=20)
            assert len(samples) == 1
            assert len(samples[0]) == 20


# ══════════════════════════════════════════════════════════════════════
# Tests: Edge cases
# ══════════════════════════════════════════════════════════════════════


class TestTimeGANBaselineEdgeCases:
    """Edge case tests."""

    def test_single_feature(self, dummy_ohlcv_multi):
        """Training with just one feature should work."""
        cfg = TimeGANBaselineConfig(
            seq_len=10, hidden_dim=4, num_layers=1,
            embedding_epochs=2, supervised_epochs=2, joint_epochs=2,
            batch_size=16, device="cpu", print_every=1,
            features=["close"],
        )
        baseline = TimeGANBaseline(cfg)
        histories = baseline.train(dummy_ohlcv_multi, tickers=["AAA"])
        assert "AAA" in baseline.trained_tickers

        samples = baseline.generate("AAA", n_samples=1, num_trading_days=30)
        assert len(samples[0]) == 30

    def test_deterministic_seed(self, dummy_ohlcv_multi, fast_config):
        """Two runs with the same seed should produce the same training."""
        import torch

        np.random.seed(42)
        torch.manual_seed(42)
        b1 = TimeGANBaseline(fast_config)
        h1 = b1.train(dummy_ohlcv_multi, tickers=["AAA"])

        np.random.seed(42)
        torch.manual_seed(42)
        b2 = TimeGANBaseline(fast_config)
        h2 = b2.train(dummy_ohlcv_multi, tickers=["AAA"])

        # Training histories should be identical
        for phase in h1.get("AAA", {}):
            if phase in h2.get("AAA", {}):
                h1_loss = h1["AAA"][phase].get("final_loss") or h1["AAA"][phase].get("final_g_loss")
                h2_loss = h2["AAA"][phase].get("final_loss") or h2["AAA"][phase].get("final_g_loss")
                if h1_loss is not None and h2_loss is not None:
                    assert abs(h1_loss - h2_loss) < 1e-4, f"Phase {phase}: {h1_loss} vs {h2_loss}"
