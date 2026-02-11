"""VPIN (Volume-Synchronized Probability of Informed Trading) computation.

Provides composable DuckDB CTEs for computing VPIN from trade data,
following the Easley, Lopez de Prado, O'Hara (2012) methodology adapted
for prediction markets with exact taker_side classification.
"""

# Minimum trades for a market to produce meaningful VPIN.
# With bucket_size=200 and lookback=10, this guarantees at least 2 complete
# VPIN windows per qualifying market.
MIN_TRADES = 500


def vpin_cte(
    trades_table: str,
    bucket_size: int = 200,
    lookback: int = 10,
) -> str:
    """DuckDB CTE chain that computes VPIN per market from a trades table.

    Expects the input table to have columns:
        ticker, count, taker_side, yes_price, created_time

    Produces a ``vpin_series`` CTE with columns:
        ticker, bucket_id, v_yes, v_no, total_vol, avg_price,
        bucket_start, bucket_end, order_imbalance, vpin, signed_flow,
        window_size

    Usage::

        WITH trades AS (...),
        {vpin_cte("trades", bucket_size=200, lookback=10)}
        SELECT * FROM vpin_series WHERE window_size = 10

    Trades are assigned to buckets by cumulative volume, not split at
    boundaries. The approximation error is negligible when bucket_size is
    much larger than individual trade sizes.
    """
    return f"""volume_running AS (
        SELECT *,
            SUM(count) OVER (
                PARTITION BY ticker ORDER BY created_time
            ) AS cum_vol
        FROM {trades_table}
    ),
    volume_buckets AS (
        SELECT *,
            FLOOR((cum_vol - 1) / {bucket_size}) AS bucket_id
        FROM volume_running
    ),
    bucket_stats AS (
        SELECT
            ticker,
            bucket_id,
            SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) AS v_yes,
            SUM(CASE WHEN taker_side = 'no' THEN count ELSE 0 END) AS v_no,
            SUM(count) AS total_vol,
            AVG(yes_price) AS avg_price,
            MIN(created_time) AS bucket_start,
            MAX(created_time) AS bucket_end
        FROM volume_buckets
        GROUP BY ticker, bucket_id
    ),
    vpin_series AS (
        SELECT
            ticker,
            bucket_id,
            v_yes,
            v_no,
            total_vol,
            avg_price,
            bucket_start,
            bucket_end,
            ABS(v_yes - v_no)::DOUBLE / total_vol AS order_imbalance,
            AVG(ABS(v_yes - v_no)::DOUBLE / total_vol) OVER w AS vpin,
            AVG((v_yes - v_no)::DOUBLE / total_vol) OVER w AS signed_flow,
            COUNT(*) OVER w AS window_size
        FROM bucket_stats
        WINDOW w AS (
            PARTITION BY ticker ORDER BY bucket_id
            ROWS BETWEEN {lookback - 1} PRECEDING AND CURRENT ROW
        )
    )"""
