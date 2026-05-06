# -*- coding: utf-8 -*-
"""
Tests for the TSTR (Train on Synthetic, Test on Real) Benchmark.

All tests use small dummy data and minimal epochs.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from sde_causal_generator.tstr_benchmark import (
    TSTRBenchmark,
    TSTRConfig,
    _build_lagged_features,
    _build_vol_features,
    _compute_auc,
    _compute_log_returns,
    _extract_returns_from_df,
    _train_classifier,
    _eval_classifier,
    _train_regressor,
    _eval_regressor,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def dummy_prices(rng) -> np.ndarray:
    """200 days of realistic close prices."""
    returns = rng.normal(0.0003, 0.015, 200)
    return 100.0 * np.exp(np.cumsum(returns))


@pytest.fixture
def dummy_returns(dummy_prices) -> np.ndarray:
    """Log returns from dummy prices."""
    return _compute_log_returns(dummy_prices)


@pytest.fixture
def dummy_train_df(rng) -> pd.DataFrame:
    """Training DataFrame with 300 rows."""
    n = 300
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame({
        "date": dates,
        "tic": "TEST",
        "close": close,
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.lognormal(15, 0.3, n).astype(int),
    })


@pytest.fixture
def dummy_test_df(rng) -> pd.DataFrame:
    """Test DataFrame with 100 rows."""
    n = 100
    dates = pd.bdate_range("2021-06-01", periods=n)
    close = 120.0 * np.exp(np.cumsum(rng.normal(0, 0.018, n)))
    return pd.DataFrame({
        "date": dates,
        "tic": "TEST",
        "close": close,
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.lognormal(15, 0.3, n).astype(int),
    })


@pytest.fixture
def dummy_synth_df(rng) -> pd.DataFrame:
    """Synthetic DataFrame with 300 rows (similar to training)."""
    n = 300
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.016, n)))
    return pd.DataFrame({
        "date": dates,
        "tic": "TEST",
        "close": close,
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.lognormal(15, 0.3, n).astype(int),
    })


@pytest.fixture
def fast_tstr_config() -> TSTRConfig:
    """Minimal TSTR config for fast testing."""
    return TSTRConfig(
        n_lags=5,
        vol_window=10,
        hidden_dim=8,
        num_layers=1,
        epochs=5,
        batch_size=32,
        patience=3,
        n_runs=1,
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════════════
# Tests: Feature engineering
# ══════════════════════════════════════════════════════════════════════


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_log_returns(self, dummy_prices):
        rets = _compute_log_returns(dummy_prices)
        assert len(rets) == len(dummy_prices) - 1
        assert np.isfinite(rets).all()

    def test_log_returns_negative_prices_clipped(self):
        """Negative prices should be clipped, not produce NaN."""
        prices = np.array([10.0, -1.0, 5.0, 8.0])
        rets = _compute_log_returns(prices)
        assert np.isfinite(rets).all()

    def test_build_lagged_features_shape(self, dummy_returns):
        X, y = _build_lagged_features(dummy_returns, n_lags=10)
        assert X.shape[1] == 10
        assert len(X) == len(y)
        assert len(X) == len(dummy_returns) - 10

    def test_build_lagged_features_labels_binary(self, dummy_returns):
        """Direction labels should be 0 or 1."""
        _, y = _build_lagged_features(dummy_returns, n_lags=5)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_build_lagged_features_empty_for_short_series(self):
        short = np.array([0.01, -0.005, 0.003])
        X, y = _build_lagged_features(short, n_lags=10)
        assert len(X) == 0
        assert len(y) == 0

    def test_build_vol_features_shape(self, dummy_returns):
        X, y = _build_vol_features(dummy_returns, n_lags=5, vol_window=10)
        assert X.shape[1] == 5
        assert len(X) == len(y)
        # Forward vol should be non-negative
        assert (y >= 0).all()

    def test_build_vol_features_values(self, dummy_returns):
        """Volatility targets should match manual computation."""
        n_lags = 5
        vol_window = 10
        X, y = _build_vol_features(dummy_returns, n_lags, vol_window)
        # Check first value manually
        expected_vol = dummy_returns[n_lags : n_lags + vol_window].std()
        np.testing.assert_almost_equal(y[0], expected_vol, decimal=6)

    def test_extract_returns_from_df(self, dummy_train_df):
        rets = _extract_returns_from_df(dummy_train_df, "TEST", "close")
        assert len(rets) == len(dummy_train_df) - 1
        assert np.isfinite(rets).all()


# ══════════════════════════════════════════════════════════════════════
# Tests: AUC computation
# ══════════════════════════════════════════════════════════════════════


class TestAUC:
    """Tests for the AUC-ROC computation."""

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _compute_auc(y_true, y_score)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_prediction(self):
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.randint(0, 2, n).astype(float)
        y_score = rng.random(n)
        auc = _compute_auc(y_true, y_score)
        # Random should be close to 0.5
        assert 0.4 < auc < 0.6

    def test_all_same_class(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        auc = _compute_auc(y_true, y_score)
        assert auc == 0.5  # undefined → default 0.5


# ══════════════════════════════════════════════════════════════════════
# Tests: Model training & evaluation
# ══════════════════════════════════════════════════════════════════════


class TestModels:
    """Tests for the classifier and regressor training."""

    def test_train_classifier(self, dummy_returns, fast_tstr_config):
        X, y = _build_lagged_features(dummy_returns, fast_tstr_config.n_lags)
        model = _train_classifier(X, y, fast_tstr_config)
        assert model is not None

    def test_eval_classifier_metrics(self, dummy_returns, fast_tstr_config):
        X, y = _build_lagged_features(dummy_returns, fast_tstr_config.n_lags)
        model = _train_classifier(X, y, fast_tstr_config)
        device = fast_tstr_config.resolve_device()
        metrics = _eval_classifier(model, X, y, device)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_regressor(self, dummy_returns, fast_tstr_config):
        X, y = _build_vol_features(
            dummy_returns, fast_tstr_config.n_lags, fast_tstr_config.vol_window
        )
        model = _train_regressor(X, y, fast_tstr_config)
        assert model is not None

    def test_eval_regressor_metrics(self, dummy_returns, fast_tstr_config):
        X, y = _build_vol_features(
            dummy_returns, fast_tstr_config.n_lags, fast_tstr_config.vol_window
        )
        model = _train_regressor(X, y, fast_tstr_config)
        device = fast_tstr_config.resolve_device()
        metrics = _eval_regressor(model, X, y, device)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0


# ══════════════════════════════════════════════════════════════════════
# Tests: TSTRBenchmark
# ══════════════════════════════════════════════════════════════════════


class TestTSTRBenchmark:
    """Integration tests for the TSTRBenchmark class."""

    def test_evaluate_single_ticker(
        self, dummy_train_df, dummy_test_df, dummy_synth_df, fast_tstr_config
    ):
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, dummy_test_df, dummy_synth_df, "TEST"
        )

        assert "direction" in results
        assert "volatility" in results
        assert "summary" in results
        assert results["ticker"] == "TEST"

    def test_direction_task_returns_trtr_and_tstr(
        self, dummy_train_df, dummy_test_df, dummy_synth_df, fast_tstr_config
    ):
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, dummy_test_df, dummy_synth_df, "TEST"
        )

        dir_res = results["direction"]
        assert "trtr" in dir_res
        assert "tstr" in dir_res
        assert "ratios" in dir_res
        assert "accuracy" in dir_res["trtr"]
        assert "accuracy" in dir_res["tstr"]

    def test_volatility_task_returns_trtr_and_tstr(
        self, dummy_train_df, dummy_test_df, dummy_synth_df, fast_tstr_config
    ):
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, dummy_test_df, dummy_synth_df, "TEST"
        )

        vol_res = results["volatility"]
        assert "trtr" in vol_res
        assert "tstr" in vol_res
        assert "mse" in vol_res["trtr"]

    def test_summary_contains_ratios(
        self, dummy_train_df, dummy_test_df, dummy_synth_df, fast_tstr_config
    ):
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, dummy_test_df, dummy_synth_df, "TEST"
        )

        summary = results["summary"]
        assert "avg_tstr_ratio" in summary
        assert "direction_accuracy_ratio" in summary
        assert "volatility_r2_ratio" in summary

    def test_evaluate_multi_ticker(self, fast_tstr_config, rng):
        """Test multi-ticker evaluation."""
        tickers = ["AAA", "BBB"]
        n = 200

        rows_train = []
        rows_test = []
        rows_synth = []

        for tic in tickers:
            dates_train = pd.bdate_range("2020-01-02", periods=n)
            close_train = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            for i, d in enumerate(dates_train):
                rows_train.append({"date": d, "tic": tic, "close": close_train[i]})
                rows_synth.append({"date": d, "tic": tic, "close": close_train[i] * (1 + rng.normal(0, 0.01))})

            dates_test = pd.bdate_range("2021-06-01", periods=80)
            close_test = 110 * np.exp(np.cumsum(rng.normal(0, 0.012, 80)))
            for i, d in enumerate(dates_test):
                rows_test.append({"date": d, "tic": tic, "close": close_test[i]})

        train_df = pd.DataFrame(rows_train)
        test_df = pd.DataFrame(rows_test)
        synth_df = pd.DataFrame(rows_synth)

        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate_multi_ticker(
            train_df, test_df, synth_df, tickers
        )

        assert "per_ticker" in results
        assert "aggregate" in results
        assert results["aggregate"]["n_tickers"] == 2

    def test_save_results(
        self, dummy_train_df, dummy_test_df, dummy_synth_df, fast_tstr_config
    ):
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, dummy_test_df, dummy_synth_df, "TEST"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "tstr_results.json")
            bench.save_results(results, output_path)

            assert os.path.exists(output_path)
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded["ticker"] == "TEST"
            assert "direction" in loaded


# ══════════════════════════════════════════════════════════════════════
# Tests: Edge cases
# ══════════════════════════════════════════════════════════════════════


class TestTSTREdgeCases:
    """Edge case tests for TSTR benchmark."""

    def test_identical_data_high_ratio(self, dummy_train_df, fast_tstr_config):
        """When synthetic = real, TSTR ratio should be close to 1."""
        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df,
            dummy_train_df,  # test = train (overfitting baseline)
            dummy_train_df,  # synth = real
            "TEST"
        )

        # With identical data, accuracy_ratio should be near 1.0
        ratio = results["summary"]["direction_accuracy_ratio"]
        # Allow wide tolerance since it's just 1 run with tiny model
        assert ratio > 0.5, f"Expected ratio > 0.5, got {ratio}"

    def test_very_short_test_data(self, dummy_train_df, dummy_synth_df, fast_tstr_config):
        """Short test data should not crash."""
        short_test = dummy_train_df.head(30).copy()
        short_test["date"] = pd.bdate_range("2021-09-01", periods=30)

        bench = TSTRBenchmark(fast_tstr_config)
        results = bench.evaluate(
            dummy_train_df, short_test, dummy_synth_df, "TEST"
        )

        # Should have results or an error message, not crash
        assert "direction" in results

    def test_reproducible_with_seed(
        self, dummy_train_df, dummy_test_df, dummy_synth_df
    ):
        """Same config with same seed should give same results."""
        cfg = TSTRConfig(
            n_lags=5, vol_window=10, hidden_dim=8, num_layers=1,
            epochs=3, batch_size=32, patience=3, n_runs=1, device="cpu",
        )

        torch.manual_seed(42)
        np.random.seed(42)
        b1 = TSTRBenchmark(cfg)
        r1 = b1.evaluate(dummy_train_df, dummy_test_df, dummy_synth_df, "TEST")

        torch.manual_seed(42)
        np.random.seed(42)
        b2 = TSTRBenchmark(cfg)
        r2 = b2.evaluate(dummy_train_df, dummy_test_df, dummy_synth_df, "TEST")

        # Accuracy should be identical
        acc1 = r1["direction"]["tstr"]["accuracy"]
        acc2 = r2["direction"]["tstr"]["accuracy"]
        assert abs(acc1 - acc2) < 1e-4
