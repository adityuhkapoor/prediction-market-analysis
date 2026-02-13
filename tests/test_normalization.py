"""Tests for platform-specific normalization CTEs.

Verifies that both Kalshi and Polymarket normalizations produce the common
schema expected by vpin_cte() and yield correct VPIN values.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis.util.normalization import (
    kalshi_normalize_cte,
    polymarket_normalize_cte,
)
from src.analysis.util.vpin import vpin_cte


@pytest.fixture
def con():
    return duckdb.connect()


# ── Kalshi normalization ─────────────────────────────────────────────────


def _make_kalshi_trades(n=2000, bias=0.5, seed=42):
    rng = np.random.default_rng(seed)
    sides = rng.choice(["yes", "no"], size=n, p=[bias, 1 - bias])
    return pd.DataFrame({
        "ticker": "TEST-MKT",
        "count": 1,
        "taker_side": sides,
        "yes_price": 50,
        "created_time": pd.date_range("2024-01-01", periods=n, freq="min"),
    })


def test_kalshi_normalization_passthrough(con):
    """Kalshi normalization should produce identical columns to raw input."""
    trades = _make_kalshi_trades()
    con.register("raw_kalshi", trades)
    cte = kalshi_normalize_cte("raw_kalshi")
    result = con.execute(f"WITH {cte} SELECT * FROM trades LIMIT 5").df()

    assert set(result.columns) == {"ticker", "count", "taker_side", "yes_price", "created_time"}
    assert result["ticker"].iloc[0] == "TEST-MKT"


def test_kalshi_normalization_produces_identical_vpin(con):
    """VPIN through Kalshi normalization should match direct vpin_cte() call."""
    trades = _make_kalshi_trades()

    # Direct VPIN
    con.register("direct_trades", trades)
    direct = con.execute(
        f"WITH {vpin_cte('direct_trades', 200, 10)} SELECT * FROM vpin_series WHERE window_size = 10"
    ).df()

    # Via normalization
    con.register("raw_kalshi", trades)
    normalized = con.execute(
        f"""WITH {kalshi_normalize_cte('raw_kalshi')},
        {vpin_cte('trades', 200, 10)}
        SELECT * FROM vpin_series WHERE window_size = 10"""
    ).df()

    assert len(direct) == len(normalized), (
        f"Row count mismatch: direct={len(direct)}, normalized={len(normalized)}"
    )
    np.testing.assert_allclose(
        direct["vpin"].values, normalized["vpin"].values,
        atol=1e-10, err_msg="VPIN values differ between direct and normalized"
    )


# ── Polymarket normalization ─────────────────────────────────────────────


@pytest.fixture
def polymarket_setup(con):
    """Set up synthetic Polymarket tables: trades, blocks, markets."""
    n = 100
    rng = np.random.default_rng(42)

    yes_token = "111"
    no_token = "222"

    # Half buys (maker_asset_id=0), half sells (maker_asset_id=token)
    maker_asset_ids = []
    taker_asset_ids = []
    maker_amounts = []
    taker_amounts = []

    for i in range(n):
        if i % 2 == 0:
            # Buy: maker provides USDC (asset_id=0), taker provides tokens
            maker_asset_ids.append("0")
            taker_asset_ids.append(yes_token)
            usdc = rng.integers(10_000, 90_000)  # USDC amount
            tokens = int(usdc / 0.5)  # at 50 cents per token
            maker_amounts.append(usdc)
            taker_amounts.append(tokens)
        else:
            # Sell: maker provides tokens, taker provides USDC
            maker_asset_ids.append(yes_token)
            taker_asset_ids.append("0")
            tokens = rng.integers(10_000, 200_000)
            usdc = int(tokens * 0.5)
            maker_amounts.append(tokens)
            taker_amounts.append(usdc)

    trades_df = pd.DataFrame({
        "block_number": np.arange(1000, 1000 + n),
        "transaction_hash": [f"0x{i:064x}" for i in range(n)],
        "log_index": range(n),
        "maker_asset_id": maker_asset_ids,
        "taker_asset_id": taker_asset_ids,
        "maker_amount": maker_amounts,
        "taker_amount": taker_amounts,
    })

    blocks_df = pd.DataFrame({
        "block_number": np.arange(1000, 1000 + n),
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="min"),
    })

    markets_df = pd.DataFrame({
        "condition_id": ["COND-001"],
        "yes_token_id": [yes_token],
        "no_token_id": [no_token],
    })

    con.register("pm_trades", trades_df)
    con.register("pm_blocks", blocks_df)
    con.register("pm_markets", markets_df)

    return {"yes_token": yes_token, "no_token": no_token}


def test_polymarket_buy_produces_yes_side(con, polymarket_setup):
    """When maker provides USDC and taker receives YES tokens → taker_side='yes'."""
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades LIMIT 1").df()

    assert set(result.columns) >= {"ticker", "count", "taker_side", "yes_price", "created_time"}
    # First row is a buy of YES tokens
    assert result["taker_side"].iloc[0] == "yes"


def test_polymarket_sell_produces_no_side(con, polymarket_setup):
    """When maker provides YES tokens and taker provides USDC → taker_side='no'.

    Selling YES tokens = bearish on YES = taker_side='no'.
    """
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    # Get second row (index 1), which is a sell
    result = con.execute(
        f"WITH {cte} SELECT * FROM trades ORDER BY created_time"
    ).df()

    # Row 1 (0-indexed): maker provides yes_token, taker provides USDC → SELL YES → taker_side='no'
    assert result["taker_side"].iloc[1] == "no"


def test_polymarket_price_conversion(con, polymarket_setup):
    """Price should be in cents (0-100), not decimal (0-1)."""
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(
        f"WITH {cte} SELECT * FROM trades ORDER BY created_time"
    ).df()

    prices = result["yes_price"]
    assert prices.min() > 0, "Prices should be positive"
    assert prices.max() <= 100, f"Prices should be <= 100 cents, got {prices.max()}"
    # Our synthetic data uses 50 cents, so prices should be near 50
    assert abs(prices.mean() - 50) < 5, f"Expected ~50 cent prices, got mean={prices.mean():.1f}"


def test_polymarket_timestamp_join(con, polymarket_setup):
    """Trades should have timestamps from the blocks table."""
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(
        f"WITH {cte} SELECT * FROM trades ORDER BY created_time"
    ).df()

    assert result["created_time"].notna().all(), "All trades should have timestamps"
    assert result["created_time"].is_monotonic_increasing, "Timestamps should be monotonic"


def test_polymarket_count_is_token_quantity(con, polymarket_setup):
    """Count should be token quantity in token units (divided by 1e6)."""
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(
        f"WITH {cte} SELECT * FROM trades ORDER BY created_time"
    ).df()

    # All counts should be positive
    assert (result["count"] > 0).all(), "All counts should be positive"
    # Token amounts in our test are ~100K range, divided by 1e6 → ~0.1 range
    # Actually, for buys: taker_amount is tokens (10000-180000), /1e6 → 0.01-0.18
    # For sells: maker_amount is tokens (10000-200000), /1e6 → 0.01-0.2
    assert result["count"].max() < 1, f"Counts should be in token units (small), got max={result['count'].max()}"


def test_polymarket_no_token_trade(con):
    """Trading NO tokens: buy NO → taker_side='no', sell NO → taker_side='yes'."""
    no_token = "999"
    yes_token = "888"

    trades_df = pd.DataFrame({
        "block_number": [2000, 2001],
        "transaction_hash": ["0xaa", "0xbb"],
        "log_index": [0, 0],
        "maker_asset_id": ["0", no_token],  # Buy NO, then Sell NO
        "taker_asset_id": [no_token, "0"],
        "maker_amount": [50000, 100000],  # USDC for buy, tokens for sell
        "taker_amount": [100000, 50000],  # tokens for buy, USDC for sell
    })

    blocks_df = pd.DataFrame({
        "block_number": [2000, 2001],
        "timestamp": pd.to_datetime(["2024-07-01 12:00", "2024-07-01 12:01"]),
    })

    markets_df = pd.DataFrame({
        "condition_id": ["COND-NO"],
        "yes_token_id": [yes_token],
        "no_token_id": [no_token],
    })

    con.register("pm_trades", trades_df)
    con.register("pm_blocks", blocks_df)
    con.register("pm_markets", markets_df)

    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(
        f"WITH {cte} SELECT * FROM trades ORDER BY created_time"
    ).df()

    assert len(result) == 2

    # Buy NO tokens → taker_side = 'no'
    assert result["taker_side"].iloc[0] == "no"
    # Sell NO tokens → taker_side = 'yes'
    assert result["taker_side"].iloc[1] == "yes"


# ── delta_vpin backward compatibility ────────────────────────────────────


def test_delta_vpin_exists_and_null_for_first_bucket(con):
    """delta_vpin column should exist and be NULL for the first bucket per market."""
    trades = _make_kalshi_trades(n=2000, bias=0.6)
    con.register("trades", trades)
    result = con.execute(
        f"WITH {vpin_cte('trades', 200, 10)} SELECT * FROM vpin_series ORDER BY bucket_id"
    ).df()

    assert "delta_vpin" in result.columns, "delta_vpin column missing from vpin_series"
    assert pd.isna(result["delta_vpin"].iloc[0]), "First bucket delta_vpin should be NULL"
    # Non-first buckets should have values
    non_null = result["delta_vpin"].dropna()
    assert len(non_null) > 0, "Should have non-null delta_vpin values after first bucket"
