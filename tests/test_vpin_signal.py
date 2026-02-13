"""Tests for VPIN signal analysis logic."""

import numpy as np
import pandas as pd

from src.analysis.kalshi.vpin_signal import VPINSignalAnalysis


def _make_vpin_df(n_markets=10, buckets_per_market=10):
    """Create a minimal vpin_df-like DataFrame for testing."""
    rows = []
    for i in range(n_markets):
        ticker = f"MARKET-{i:03d}"
        result = "yes" if i % 2 == 0 else "no"
        for b in range(buckets_per_market):
            rows.append({
                "ticker": ticker,
                "bucket_id": b,
                "vpin": 0.3 + 0.02 * b,
                "signed_flow": 0.1 if result == "yes" else -0.1,
                "avg_price": 50 + b,
                "result": result,
                "event_ticker": f"EVENT-{i:03d}",
                "close_time": pd.Timestamp("2024-06-01") + pd.Timedelta(days=i),
                "window_size": 10,
                "order_imbalance": 0.3,
                "v_yes": 120,
                "v_no": 80,
                "total_vol": 200,
                "bucket_start": pd.Timestamp("2024-01-01"),
                "bucket_end": pd.Timestamp("2024-01-02"),
            })
    return pd.DataFrame(rows)


def test_first_half_truncation():
    """Truncation logic should keep only the first half of each market's buckets."""
    df = _make_vpin_df(n_markets=5, buckets_per_market=10)

    bucket_counts = df.groupby("ticker")["bucket_id"].transform("count")
    bucket_rank = df.groupby("ticker")["bucket_id"].rank(method="first")
    early_df = df[bucket_rank <= (bucket_counts / 2)].copy()

    # Each market has 10 buckets, first half = 5
    for ticker in df["ticker"].unique():
        n_early = len(early_df[early_df["ticker"] == ticker])
        assert n_early == 5, f"Expected 5 early buckets for {ticker}, got {n_early}"


def test_min_markets_guard():
    """With fewer than MIN_MARKETS markets, should return limited results dict."""
    analysis = VPINSignalAnalysis.__new__(VPINSignalAnalysis)
    analysis.name = "vpin_signal"

    # Create df with only 10 markets (below MIN_MARKETS=50)
    df = _make_vpin_df(n_markets=10, buckets_per_market=20)
    result = analysis._test_resolution_prediction(df)

    assert "n_markets" in result, "Should have n_markets key"
    assert "baseline_brier_mean" in result, "Should have baseline_brier_mean"
    # Should NOT have model_brier (not enough markets to train)
    assert "model_brier" not in result, "Should not train model with < 50 markets"


def test_fdr_keys_present():
    """After FDR correction, results should contain *_pvalue_fdr_* keys."""
    analysis = VPINSignalAnalysis.__new__(VPINSignalAnalysis)
    analysis.name = "vpin_signal"

    results = {
        "volatility": {
            "vol_pvalue_k1": 0.01,
            "vol_pvalue_k5": 0.001,
            "vol_pvalue_k10": 1e-7,
            "vol_beta_k1": 0.1,
        },
        "direction": {
            "dir_pvalue_k1": 0.06,
            "dir_pvalue_k5": 0.05,
            "dir_pvalue_k10": 0.03,
            "dir_beta_k1": 0.04,
        },
    }
    analysis._apply_fdr_correction(results)

    for test_name in ("volatility", "direction"):
        for key in results[test_name]:
            if "pvalue_fdr" in key:
                return  # Found at least one FDR key
    raise AssertionError("No FDR-corrected p-value keys found in results")


def test_fdr_values_geq_raw():
    """Every FDR-adjusted p-value should be >= its corresponding raw p-value."""
    analysis = VPINSignalAnalysis.__new__(VPINSignalAnalysis)
    analysis.name = "vpin_signal"

    results = {
        "volatility": {
            "vol_pvalue_k1": 0.01,
            "vol_pvalue_k5": 0.001,
            "vol_pvalue_k10": 1e-7,
        },
        "direction": {
            "dir_pvalue_k1": 0.06,
            "dir_pvalue_k5": 0.05,
            "dir_pvalue_k10": 0.03,
        },
    }
    analysis._apply_fdr_correction(results)

    for test_name in ("volatility", "direction"):
        for key, val in results[test_name].items():
            if "pvalue_fdr" not in key:
                continue
            raw_key = key.replace("pvalue_fdr", "pvalue")
            raw_val = results[test_name][raw_key]
            assert val >= raw_val, (
                f"{test_name}.{key}={val} < raw {raw_key}={raw_val}"
            )
