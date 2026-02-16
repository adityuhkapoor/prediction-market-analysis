from src.analysis.util.vpin import vpin_cte


def run_vpin(con, trades_df, bucket_size=200, lookback=10):
    con.register("trades", trades_df)
    return con.execute(
        f"""
        WITH {vpin_cte("trades", bucket_size, lookback)}
        SELECT * FROM vpin_series
        WHERE window_size = {lookback}
        """
    ).df()


def test_balanced_vpin_near_zero(con, synthetic_trades):
    trades = synthetic_trades(n_trades=2000, bias=0.5)
    result = run_vpin(con, trades)
    assert len(result) > 0
    assert result["vpin"].mean() < 0.15


def test_imbalanced_vpin_is_one(con, synthetic_trades):
    trades = synthetic_trades(n_trades=2000, bias=1.0)
    result = run_vpin(con, trades)
    assert len(result) > 0
    assert result["vpin"].iloc[-1] == 1.0


def test_signed_flow_positive_for_yes_bias(con, synthetic_trades):
    trades = synthetic_trades(n_trades=2000, bias=0.8)
    result = run_vpin(con, trades)
    assert result["signed_flow"].mean() > 0


def test_signed_flow_negative_for_no_bias(con, synthetic_trades):
    trades = synthetic_trades(n_trades=2000, bias=0.2)
    result = run_vpin(con, trades)
    assert result["signed_flow"].mean() < 0


def test_window_size_ramps(con, synthetic_trades):
    """Window starts at 1 and ramps to lookback."""
    trades = synthetic_trades(n_trades=2000, bias=0.5)
    con.register("trades", trades)
    result = con.execute(
        f"""
        WITH {vpin_cte("trades", 200, 10)}
        SELECT * FROM vpin_series
        ORDER BY bucket_id
        """
    ).df()
    assert result["window_size"].iloc[0] == 1
    assert result["window_size"].max() == 10


def test_bucket_id_assignment(con, synthetic_trades):
    trades = synthetic_trades(n_trades=600, bias=0.5)
    con.register("trades", trades)
    result = con.execute(
        f"""
        WITH {vpin_cte("trades", 200, 10)}
        SELECT * FROM vpin_series
        ORDER BY bucket_id
        """
    ).df()
    bucket_ids = sorted(result["bucket_id"].unique())
    assert bucket_ids == [0, 1, 2]
