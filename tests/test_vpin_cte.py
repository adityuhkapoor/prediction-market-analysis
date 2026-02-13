"""Tests for VPIN CTE computation on synthetic data."""

from src.analysis.util.vpin import vpin_cte


def _run_vpin(con, trades_df, bucket_size=200, lookback=10):
    """Register trades DataFrame and execute VPIN CTE, return full-window rows."""
    con.register("trades", trades_df)
    return con.execute(
        f"""
        WITH {vpin_cte("trades", bucket_size, lookback)}
        SELECT * FROM vpin_series
        WHERE window_size = {lookback}
        """
    ).df()


def test_balanced_vpin_near_zero(con, synthetic_trades):
    """Balanced flow (50/50) should produce VPIN near 0."""
    trades = synthetic_trades(n_trades=2000, bias=0.5)
    result = _run_vpin(con, trades)
    assert len(result) > 0, "No VPIN rows produced"
    # Full-window VPIN should be low for balanced flow
    assert result["vpin"].mean() < 0.15, f"VPIN too high for balanced flow: {result['vpin'].mean():.4f}"


def test_imbalanced_vpin_near_one(con, synthetic_trades):
    """Fully imbalanced flow (100% yes) should produce VPIN = 1.0."""
    trades = synthetic_trades(n_trades=2000, bias=1.0)
    result = _run_vpin(con, trades)
    assert len(result) > 0, "No VPIN rows produced"
    assert result["vpin"].iloc[-1] == 1.0, f"Expected VPIN=1.0 for pure yes flow, got {result['vpin'].iloc[-1]}"


def test_signed_flow_positive_for_yes_bias(con, synthetic_trades):
    """Yes-biased flow should produce positive signed flow."""
    trades = synthetic_trades(n_trades=2000, bias=0.8)
    result = _run_vpin(con, trades)
    assert result["signed_flow"].mean() > 0, (
        f"Expected positive signed_flow for yes bias, got {result['signed_flow'].mean():.4f}"
    )


def test_signed_flow_negative_for_no_bias(con, synthetic_trades):
    """No-biased flow should produce negative signed flow."""
    trades = synthetic_trades(n_trades=2000, bias=0.2)
    result = _run_vpin(con, trades)
    assert result["signed_flow"].mean() < 0, (
        f"Expected negative signed_flow for no bias, got {result['signed_flow'].mean():.4f}"
    )


def test_window_size_ramps(con, synthetic_trades):
    """Window size should start at 1 and ramp up to the lookback value."""
    trades = synthetic_trades(n_trades=2000, bias=0.5)
    con.register("trades", trades)
    # Get ALL rows (not just full-window)
    result = con.execute(
        f"""
        WITH {vpin_cte("trades", 200, 10)}
        SELECT * FROM vpin_series
        ORDER BY bucket_id
        """
    ).df()
    assert result["window_size"].iloc[0] == 1, "First bucket should have window_size=1"
    assert result["window_size"].max() == 10, "Should reach lookback window_size=10"


def test_bucket_id_assignment(con, synthetic_trades):
    """With count=1, bucket_id should be floor((row_num-1)/bucket_size)."""
    trades = synthetic_trades(n_trades=600, bias=0.5)
    con.register("trades", trades)
    result = con.execute(
        f"""
        WITH {vpin_cte("trades", 200, 10)}
        SELECT * FROM vpin_series
        ORDER BY bucket_id
        """
    ).df()
    # 600 trades with count=1 and bucket_size=200 → bucket_ids 0, 1, 2
    bucket_ids = sorted(result["bucket_id"].unique())
    assert bucket_ids == [0, 1, 2], f"Expected buckets [0, 1, 2], got {bucket_ids}"
