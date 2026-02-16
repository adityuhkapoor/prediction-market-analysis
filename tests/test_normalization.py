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


def fake_kalshi_trades(n=2000, bias=0.5, seed=42):
    rng = np.random.default_rng(seed)
    sides = rng.choice(["yes", "no"], size=n, p=[bias, 1 - bias])
    return pd.DataFrame(
        {
            "ticker": "TEST-MKT",
            "count": 1,
            "taker_side": sides,
            "yes_price": 50,
            "created_time": pd.date_range("2024-01-01", periods=n, freq="min"),
        }
    )


def test_kalshi_passthrough(con):
    trades = fake_kalshi_trades()
    con.register("raw_kalshi", trades)
    cte = kalshi_normalize_cte("raw_kalshi")
    result = con.execute(f"WITH {cte} SELECT * FROM trades LIMIT 5").df()

    assert set(result.columns) == {"ticker", "count", "taker_side", "yes_price", "created_time"}
    assert result["ticker"].iloc[0] == "TEST-MKT"


def test_kalshi_vpin_matches_direct(con):
    """Normalization is a passthrough so VPIN should be identical either way."""
    trades = fake_kalshi_trades()

    con.register("direct_trades", trades)
    direct = con.execute(
        f"WITH {vpin_cte('direct_trades', 200, 10)} SELECT * FROM vpin_series WHERE window_size = 10"
    ).df()

    con.register("raw_kalshi", trades)
    normalized = con.execute(
        f"""WITH {kalshi_normalize_cte("raw_kalshi")},
        {vpin_cte("trades", 200, 10)}
        SELECT * FROM vpin_series WHERE window_size = 10"""
    ).df()

    assert len(direct) == len(normalized)
    np.testing.assert_allclose(direct["vpin"].values, normalized["vpin"].values, atol=1e-10)


@pytest.fixture
def polymarket_setup(con):
    n = 100
    rng = np.random.default_rng(42)

    yes_token = "111"
    no_token = "222"

    maker_asset_ids = []
    taker_asset_ids = []
    maker_amounts = []
    taker_amounts = []

    for i in range(n):
        if i % 2 == 0:
            maker_asset_ids.append("0")  # buy: maker provides USDC
            taker_asset_ids.append(yes_token)
            usdc = rng.integers(10_000, 90_000)
            tokens = int(usdc / 0.5)
            maker_amounts.append(usdc)
            taker_amounts.append(tokens)
        else:
            maker_asset_ids.append(yes_token)  # sell: maker provides tokens
            taker_asset_ids.append("0")
            tokens = rng.integers(10_000, 200_000)
            usdc = int(tokens * 0.5)
            maker_amounts.append(tokens)
            taker_amounts.append(usdc)

    trades_df = pd.DataFrame(
        {
            "block_number": np.arange(1000, 1000 + n),
            "transaction_hash": [f"0x{i:064x}" for i in range(n)],
            "log_index": range(n),
            "maker_asset_id": maker_asset_ids,
            "taker_asset_id": taker_asset_ids,
            "maker_amount": maker_amounts,
            "taker_amount": taker_amounts,
        }
    )

    blocks_df = pd.DataFrame(
        {
            "block_number": np.arange(1000, 1000 + n),
            "timestamp": pd.date_range("2024-06-01", periods=n, freq="min"),
        }
    )

    markets_df = pd.DataFrame(
        {
            "condition_id": ["COND-001"],
            "yes_token_id": [yes_token],
            "no_token_id": [no_token],
        }
    )

    con.register("pm_trades", trades_df)
    con.register("pm_blocks", blocks_df)
    con.register("pm_markets", markets_df)

    return {"yes_token": yes_token, "no_token": no_token}


def test_polymarket_buy_yes_side(con, polymarket_setup):
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades LIMIT 1").df()

    assert set(result.columns) >= {"ticker", "count", "taker_side", "yes_price", "created_time"}
    assert result["taker_side"].iloc[0] == "yes"


def test_polymarket_sell_no_side(con, polymarket_setup):
    """Selling YES tokens is bearish → taker_side='no'."""
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades ORDER BY created_time").df()
    assert result["taker_side"].iloc[1] == "no"


def test_polymarket_price_in_cents(con, polymarket_setup):
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades ORDER BY created_time").df()

    prices = result["yes_price"]
    assert prices.min() > 0
    assert prices.max() <= 100
    assert abs(prices.mean() - 50) < 5


def test_polymarket_timestamps(con, polymarket_setup):
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades ORDER BY created_time").df()
    assert result["created_time"].notna().all()
    assert result["created_time"].is_monotonic_increasing


def test_polymarket_token_quantity(con, polymarket_setup):
    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades ORDER BY created_time").df()

    assert (result["count"] > 0).all()
    assert result["count"].max() < 1  # tokens / 1e6


def test_polymarket_no_token_sides(con):
    """Buy NO → 'no', sell NO → 'yes'."""
    no_token = "999"
    yes_token = "888"

    trades_df = pd.DataFrame(
        {
            "block_number": [2000, 2001],
            "transaction_hash": ["0xaa", "0xbb"],
            "log_index": [0, 0],
            "maker_asset_id": ["0", no_token],
            "taker_asset_id": [no_token, "0"],
            "maker_amount": [50000, 100000],
            "taker_amount": [100000, 50000],
        }
    )

    blocks_df = pd.DataFrame(
        {
            "block_number": [2000, 2001],
            "timestamp": pd.to_datetime(["2024-07-01 12:00", "2024-07-01 12:01"]),
        }
    )

    markets_df = pd.DataFrame(
        {
            "condition_id": ["COND-NO"],
            "yes_token_id": [yes_token],
            "no_token_id": [no_token],
        }
    )

    con.register("pm_trades", trades_df)
    con.register("pm_blocks", blocks_df)
    con.register("pm_markets", markets_df)

    cte = polymarket_normalize_cte("pm_trades", "pm_blocks", "pm_markets")
    result = con.execute(f"WITH {cte} SELECT * FROM trades ORDER BY created_time").df()

    assert len(result) == 2
    assert result["taker_side"].iloc[0] == "no"
    assert result["taker_side"].iloc[1] == "yes"


def test_delta_vpin_first_bucket_null(con):
    trades = fake_kalshi_trades(n=2000, bias=0.6)
    con.register("trades", trades)
    result = con.execute(f"WITH {vpin_cte('trades', 200, 10)} SELECT * FROM vpin_series ORDER BY bucket_id").df()

    assert "delta_vpin" in result.columns
    assert pd.isna(result["delta_vpin"].iloc[0])
    assert result["delta_vpin"].dropna().shape[0] > 0
