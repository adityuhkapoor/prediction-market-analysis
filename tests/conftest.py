import duckdb
import pandas as pd
import pytest


@pytest.fixture
def con():
    return duckdb.connect()


@pytest.fixture
def synthetic_trades():
    def _make(n_trades=1000, ticker="TEST-TICKER", bucket_size=200, bias=0.5):
        import numpy as np
        rng = np.random.default_rng(42)
        sides = rng.choice(["yes", "no"], size=n_trades, p=[bias, 1 - bias])
        return pd.DataFrame({
            "ticker": ticker,
            "count": 1,
            "taker_side": sides,
            "yes_price": 50,
            "created_time": pd.date_range("2024-01-01", periods=n_trades, freq="min"),
        })
    return _make
