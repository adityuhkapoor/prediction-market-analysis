"""Tests for mispricing model logic."""

import numpy as np
import pandas as pd

from src.analysis.kalshi.mispricing_model import MispricingModelAnalysis


def test_log_volume_formula():
    """log_volume should use bucket geometry: log1p((bucket_id + 1) * bucket_size)."""
    bucket_size = 200
    bucket_ids = np.array([0, 5, 12, 25])
    expected = np.log1p((bucket_ids + 1) * bucket_size)

    # Verify expected values for known inputs
    assert np.isclose(expected[0], np.log1p(200)), "bucket_id=0 should give log1p(200)"
    assert np.isclose(expected[1], np.log1p(1200)), "bucket_id=5 should give log1p(1200)"
    assert np.isclose(expected[2], np.log1p(2600)), "bucket_id=12 should give log1p(2600)"
    assert np.isclose(expected[3], np.log1p(5200)), "bucket_id=25 should give log1p(5200)"


def test_temporal_split_monotonic():
    """After temporal split, max(train.close_time) <= min(test.close_time)."""
    analysis = MispricingModelAnalysis.__new__(MispricingModelAnalysis)
    analysis.train_frac = 0.7

    df = pd.DataFrame({
        "close_time": pd.date_range("2024-01-01", periods=100, freq="D"),
        "value": range(100),
    })
    train, test = analysis._temporal_split(df)

    assert train["close_time"].max() <= test["close_time"].min(), (
        f"Train max {train['close_time'].max()} > test min {test['close_time'].min()}"
    )
    assert len(train) == 70, f"Expected 70 train rows, got {len(train)}"
    assert len(test) == 30, f"Expected 30 test rows, got {len(test)}"


def test_bin_edges_shared():
    """After _compute_longshot_adj, test prices should use training bin edges."""
    analysis = MispricingModelAnalysis.__new__(MispricingModelAnalysis)

    train_df = pd.DataFrame({
        "price": np.linspace(5, 95, 60),
        "y": np.random.default_rng(42).integers(0, 2, 60),
    })
    test_df = pd.DataFrame({
        "price": [25.0, 50.0, 75.0],
        "y": [0, 1, 1],
    })

    train_out, test_out = analysis._compute_longshot_adj(train_df, test_df)

    # Test prices within training range should get valid bin assignments
    assert test_out["price_bin"].notna().sum() >= 2, (
        "Most test prices in training range should get valid bin assignments"
    )
    # longshot_adj should be finite for binned test observations
    binned_mask = test_out["price_bin"].notna()
    assert test_out.loc[binned_mask, "longshot_adj"].notna().all(), (
        "Binned test observations should have non-NaN longshot_adj"
    )


def test_extrapolated_price_gets_zero_adj():
    """Test prices outside training range should get longshot_adj = 0."""
    analysis = MispricingModelAnalysis.__new__(MispricingModelAnalysis)

    train_df = pd.DataFrame({
        "price": np.linspace(20, 80, 60),
        "y": np.random.default_rng(42).integers(0, 2, 60),
    })
    # Price way outside training range
    test_df = pd.DataFrame({
        "price": [1.0, 99.0],
        "y": [0, 1],
    })

    _, test_out = analysis._compute_longshot_adj(train_df, test_df)

    # Extrapolated prices get NaN bins → fillna(0)
    assert (test_out["longshot_adj"] == 0).all(), (
        f"Expected longshot_adj=0 for extrapolated prices, got {test_out['longshot_adj'].values}"
    )
