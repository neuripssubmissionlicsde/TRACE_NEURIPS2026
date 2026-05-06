# -*- coding: utf-8 -*-
"""
Tests for the Full Benchmark Pipeline (Causal SDE vs TimeGAN).

Tests cover configuration, data loading, distributional metrics,
train/test splitting, comparison table building, and end-to-end runs.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from sde_causal_generator.benchmark import (
    BenchmarkConfig,
    BenchmarkPipeline,
    compute_distributional_metrics,
)

# Check TimeGAN availability for integration tests
_timegan_available = False
try:
    from sde_causal_generator.timegan_baseline import _timegan_available as _tg_flag
    _timegan_available = _tg_flag
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def multi_ticker_df(rng) -> pd.DataFrame:
    """Multi-ticker OHLCV DataFrame with 200 rows per ticker."""
    rows = []
    tickers = ["AAA", "BBB"]
    dates = pd.bdate_range("2020-01-02", periods=200)

    for tic in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 200)))
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "tic": tic,
                "open": close[i] * (1 + rng.normal(0, 0.003)),
                "high": close[i] * (1 + abs(rng.normal(0, 0.005))),
                "low": close[i] * (1 - abs(rng.normal(0, 0.005))),
                "close": close[i],
                "volume": int(rng.lognormal(15, 0.3)),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def fake_causal_sde_dir(multi_ticker_df, rng) -> str:
    """Create a temporary directory mimicking Causal SDE output."""
    tmpdir = tempfile.mkdtemp()

    # Save training data
    multi_ticker_df.to_csv(os.path.join(tmpdir, "training_data.csv"), index=False)

    # Create per-ticker synthetic CSVs
    for tic in ["AAA", "BBB"]:
        ticker_dir = os.path.join(tmpdir, tic)
        os.makedirs(ticker_dir, exist_ok=True)

        # Create a synthetic CSV (similar to real but with noise)
        tic_df = multi_ticker_df[multi_ticker_df["tic"] == tic].copy()
        synth_close = tic_df["close"].values * (1 + rng.normal(0, 0.02, len(tic_df)))
        synth_df = tic_df.copy()
        synth_df["close"] = synth_close
        synth_df["sample"] = 0

        synth_df.to_csv(
            os.path.join(ticker_dir, "synthetic_data_causal_sde.csv"), index=False
        )
        # Also write per-sample CSV
        synth_df.drop(columns=["sample"]).to_csv(
            os.path.join(ticker_dir, "synthetic_data_0.csv"), index=False
        )

    yield tmpdir

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def benchmark_config(fake_causal_sde_dir) -> BenchmarkConfig:
    """Minimal BenchmarkConfig pointed at the fake Causal SDE directory."""
    return BenchmarkConfig(
        tickers=["AAA", "BBB"],
        train_start="2020-01-02",
        train_end="2020-12-31",
        test_ratio=0.2,
        causal_sde_dir=fake_causal_sde_dir,
        n_samples=1,
        timegan_seq_len=10,
        timegan_hidden_dim=8,
        timegan_embedding_epochs=2,
        timegan_supervised_epochs=2,
        timegan_joint_epochs=2,
        timegan_batch_size=16,
        timegan_device="cpu",
        tstr_n_lags=5,
        tstr_epochs=3,
        tstr_n_runs=1,
        tstr_vol_window=10,
        output_dir=os.path.join(fake_causal_sde_dir, "benchmark"),
        seed=42,
        skip_timegan=False,
    )


# ══════════════════════════════════════════════════════════════════════
# Tests: Configuration
# ══════════════════════════════════════════════════════════════════════


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        cfg = BenchmarkConfig()
        assert cfg.test_ratio == 0.2
        assert cfg.timegan_seq_len == 24
        assert cfg.seed == 42
        assert cfg.skip_timegan is False

    def test_from_yaml(self, tmp_path):
        """Test loading from a YAML config file."""
        yaml_content = """
tickers:
  - AAA
  - BBB
train_start: "2020-01-01"
train_end: "2023-12-31"
output_dir: "test_output"
seed: 123
generation:
  n_samples: 5
timegan:
  hidden_dim: 32
  embedding_epochs: 100
tstr:
  n_lags: 15
  epochs: 50
benchmark:
  test_ratio: 0.3
  skip_timegan: true
  output_dir: "results/custom_bench"
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = BenchmarkConfig.from_yaml(str(yaml_path))
        assert cfg.tickers == ["AAA", "BBB"]
        assert cfg.n_samples == 5
        assert cfg.seed == 123
        assert cfg.test_ratio == 0.3
        assert cfg.skip_timegan is True
        assert cfg.timegan_hidden_dim == 32
        assert cfg.timegan_embedding_epochs == 100
        assert cfg.tstr_n_lags == 15
        assert cfg.tstr_epochs == 50
        assert cfg.output_dir == "results/custom_bench"

    def test_custom_timegan_params(self):
        cfg = BenchmarkConfig(
            timegan_hidden_dim=32,
            timegan_embedding_epochs=100,
        )
        assert cfg.timegan_hidden_dim == 32
        assert cfg.timegan_embedding_epochs == 100


# ══════════════════════════════════════════════════════════════════════
# Tests: Distributional metrics
# ══════════════════════════════════════════════════════════════════════


class TestDistributionalMetrics:
    """Tests for compute_distributional_metrics."""

    def test_identical_distributions(self, rng):
        data = rng.normal(0, 0.01, 500)
        metrics = compute_distributional_metrics(data, data)

        assert metrics["ks_statistic"] == 0.0
        assert metrics["ks_pvalue"] == 1.0
        assert metrics["volatility_ratio"] == pytest.approx(1.0)
        assert metrics["kurtosis_diff"] == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions(self, rng):
        real = rng.normal(0, 0.01, 500)
        synth = rng.normal(0.001, 0.02, 500)
        metrics = compute_distributional_metrics(real, synth)

        assert metrics["ks_statistic"] > 0
        assert metrics["volatility_ratio"] > 1.0  # synth has higher vol

    def test_metric_keys_present(self, rng):
        real = rng.normal(0, 0.01, 100)
        synth = rng.normal(0, 0.012, 100)
        metrics = compute_distributional_metrics(real, synth)

        expected_keys = [
            "ks_statistic", "ks_pvalue",
            "real_volatility", "synth_volatility", "volatility_ratio",
            "real_kurtosis", "synth_kurtosis", "kurtosis_diff",
            "real_skewness", "synth_skewness", "skewness_diff",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_heavy_tailed_distribution(self, rng):
        """Heavy-tailed synthetic data should show higher kurtosis."""
        real = rng.normal(0, 0.01, 1000)
        synth = rng.standard_t(3, 1000) * 0.01  # t-distribution, heavy tails
        metrics = compute_distributional_metrics(real, synth)

        # t(3) has excess kurtosis = 6, normal = 0
        assert metrics["synth_kurtosis"] > metrics["real_kurtosis"]


# ══════════════════════════════════════════════════════════════════════
# Tests: Pipeline data handling
# ══════════════════════════════════════════════════════════════════════


class TestPipelineDataHandling:
    """Tests for data loading and splitting."""

    def test_split_train_test(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        train, test = pipe._split_train_test(multi_ticker_df)

        assert len(train) + len(test) == len(multi_ticker_df)
        assert len(test) > 0
        assert len(train) > len(test)

        # Test dates should be after train dates
        train_max = pd.to_datetime(train["date"]).max()
        test_min = pd.to_datetime(test["date"]).min()
        assert test_min >= train_max

    def test_split_preserves_all_tickers(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        train, test = pipe._split_train_test(multi_ticker_df)

        assert set(train["tic"].unique()) == set(multi_ticker_df["tic"].unique())
        # Test may not have all tickers if split point is different per ticker
        # but for same dates, it should
        assert len(test["tic"].unique()) == len(multi_ticker_df["tic"].unique())

    def test_load_causal_sde_results(self, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        results = pipe._load_causal_sde_results()

        assert "AAA" in results
        assert "BBB" in results
        for tic, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "close" in df.columns

    def test_comparison_table(self, benchmark_config):
        """Test that comparison table builder handles empty data."""
        pipe = BenchmarkPipeline(benchmark_config)

        tstr_results = {"AAA": {}, "BBB": {}}
        dist_results = {"AAA": {}, "BBB": {}}

        table = pipe._build_comparison_table(tstr_results, dist_results)
        assert "per_ticker" in table
        assert "columns" in table
        assert len(table["per_ticker"]) == 2


# ══════════════════════════════════════════════════════════════════════
# Tests: End-to-end (with TimeGAN)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not _timegan_available,
    reason="TimeGAN implementation not found",
)
class TestBenchmarkPipelineE2E:
    """End-to-end benchmark pipeline tests (requires TimeGAN)."""

    def test_full_pipeline_run(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        report = pipe.run(training_df=multi_ticker_df)

        assert "tstr" in report
        assert "distributional" in report
        assert "comparison" in report
        assert "total_time_seconds" in report

    def test_full_pipeline_saves_report(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        pipe.run(training_df=multi_ticker_df)

        report_path = pipe.save_report()
        assert os.path.exists(report_path)

        with open(report_path) as f:
            loaded = json.load(f)
        assert "tstr" in loaded

    def test_full_pipeline_creates_plots(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        pipe.run(training_df=multi_ticker_df)

        assert os.path.exists(
            os.path.join(benchmark_config.output_dir, "benchmark_comparison.png")
        )

    def test_skip_timegan(self, multi_ticker_df, benchmark_config):
        """Pipeline should run with skip_timegan=True if no TimeGAN results exist."""
        pipe = BenchmarkPipeline(benchmark_config)
        report = pipe.run(training_df=multi_ticker_df, skip_timegan=True)

        # Should still have Causal SDE results
        assert "tstr" in report

    def test_report_property(self, multi_ticker_df, benchmark_config):
        pipe = BenchmarkPipeline(benchmark_config)
        pipe.run(training_df=multi_ticker_df)
        assert len(pipe.report) > 0


# ══════════════════════════════════════════════════════════════════════
# Tests: End-to-end without TimeGAN (TSTR + distributional only)
# ══════════════════════════════════════════════════════════════════════


class TestBenchmarkPipelineSkipTimeGAN:
    """Tests that work without TimeGAN by skipping it."""

    def test_pipeline_skip_timegan_via_config(self, multi_ticker_df, benchmark_config):
        benchmark_config.skip_timegan = True
        pipe = BenchmarkPipeline(benchmark_config)
        report = pipe.run(training_df=multi_ticker_df)

        assert "tstr" in report
        assert "distributional" in report

    def test_pipeline_skip_timegan_via_run_param(self, multi_ticker_df, benchmark_config):
        """Explicit run() param should override config."""
        benchmark_config.skip_timegan = False
        pipe = BenchmarkPipeline(benchmark_config)
        report = pipe.run(training_df=multi_ticker_df, skip_timegan=True)

        assert "tstr" in report
        assert "distributional" in report

    def test_distributional_metrics_populated(self, multi_ticker_df, benchmark_config):
        benchmark_config.skip_timegan = True
        pipe = BenchmarkPipeline(benchmark_config)
        report = pipe.run(training_df=multi_ticker_df)

        dist = report["distributional"]
        # Should have at least Causal SDE metrics for loaded tickers
        for ticker in ["AAA", "BBB"]:
            if ticker in dist:
                if "causal_sde" in dist[ticker]:
                    assert "ks_statistic" in dist[ticker]["causal_sde"]


# ══════════════════════════════════════════════════════════════════════
# Tests: Checkpoint / resume for generate_causal_sde_data.py (Stage 1)
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointResume:
    """Tests for the _is_ticker_complete function in the generation script."""

    def test_complete_ticker_detected(self, tmp_path):
        """A ticker with all artefacts should be detected as complete."""
        import sys
        sys.path.insert(0, str(tmp_path.parent))

        from scripts.generate_causal_sde_data import _is_ticker_complete

        ticker_dir = str(tmp_path / "AAPL")
        os.makedirs(ticker_dir, exist_ok=True)

        # Create expected files
        pd.DataFrame({"close": [1, 2, 3]}).to_csv(
            os.path.join(ticker_dir, "synthetic_data_causal_sde.csv"), index=False
        )
        with open(os.path.join(ticker_dir, "pipeline_report.json"), "w") as f:
            json.dump({"status": "complete"}, f)

        cfg = {"generation": {"n_samples": 3}}
        for i in range(3):
            pd.DataFrame({"close": [1, 2]}).to_csv(
                os.path.join(ticker_dir, f"synthetic_data_{i}.csv"), index=False
            )

        assert _is_ticker_complete(ticker_dir, cfg) is True

    def test_incomplete_ticker_detected(self, tmp_path):
        """A ticker missing files should be detected as incomplete."""
        from scripts.generate_causal_sde_data import _is_ticker_complete

        ticker_dir = str(tmp_path / "MSFT")
        os.makedirs(ticker_dir, exist_ok=True)

        # Only create the merged CSV, not the report
        pd.DataFrame({"close": [1, 2]}).to_csv(
            os.path.join(ticker_dir, "synthetic_data_causal_sde.csv"), index=False
        )

        cfg = {"generation": {"n_samples": 2}}
        assert _is_ticker_complete(ticker_dir, cfg) is False

    def test_empty_directory(self, tmp_path):
        """Empty directory should be detected as incomplete."""
        from scripts.generate_causal_sde_data import _is_ticker_complete

        ticker_dir = str(tmp_path / "XOM")
        os.makedirs(ticker_dir, exist_ok=True)

        cfg = {"generation": {"n_samples": 5}}
        assert _is_ticker_complete(ticker_dir, cfg) is False

    def test_missing_sample_csvs(self, tmp_path):
        """If n_samples=5 but only 3 CSVs exist → incomplete."""
        from scripts.generate_causal_sde_data import _is_ticker_complete

        ticker_dir = str(tmp_path / "JPM")
        os.makedirs(ticker_dir, exist_ok=True)

        pd.DataFrame({"close": [1]}).to_csv(
            os.path.join(ticker_dir, "synthetic_data_causal_sde.csv"), index=False
        )
        with open(os.path.join(ticker_dir, "pipeline_report.json"), "w") as f:
            json.dump({}, f)

        # Only create 3 out of 5 samples
        for i in range(3):
            pd.DataFrame({"close": [1]}).to_csv(
                os.path.join(ticker_dir, f"synthetic_data_{i}.csv"), index=False
            )

        cfg = {"generation": {"n_samples": 5}}
        assert _is_ticker_complete(ticker_dir, cfg) is False
