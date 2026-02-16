"""VPIN computation as composable DuckDB CTEs (Easley, Lopez de Prado, O'Hara 2012)."""

# ≥2 full VPIN windows at bucket_size=200, lookback=10
MIN_TRADES = 500


def qualified_trades_cte(
    trades_dir: str,
    markets_dir: str,
    min_trades: int = MIN_TRADES,
) -> str:
    """Filter to finalized markets with sufficient volume.

    Produces CTEs: market_info, qualified_markets, trades.

    Usage::

        f\"\"\"
        WITH {qualified_trades_cte(trades_dir, markets_dir)},
        {vpin_cte("trades", bucket_size, lookback)}
        SELECT ...
        \"\"\"
    """
    return f"""market_info AS (
        SELECT ticker, event_ticker, result, close_time
        FROM '{markets_dir}/*.parquet'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    qualified_markets AS (
        SELECT m.ticker
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN market_info m ON t.ticker = m.ticker
        GROUP BY m.ticker
        HAVING SUM(t.count) >= {min_trades}
    ),
    trades AS (
        SELECT t.ticker, t.count, t.taker_side, t.yes_price, t.created_time
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN qualified_markets q ON t.ticker = q.ticker
    )"""


def vpin_cte(
    trades_table: str,
    bucket_size: int = 200,
    lookback: int = 10,
) -> str:
    """Compute VPIN per market from a trades table.

    Input columns: ticker, count, taker_side, yes_price, created_time

    Output CTE ``vpin_series``: ticker, bucket_id, v_yes, v_no, total_vol,
    avg_price, bucket_start, bucket_end, order_imbalance, vpin, signed_flow,
    window_size, delta_vpin
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
    vpin_raw AS (
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
    ),
    vpin_series AS (
        SELECT
            *,
            vpin - LAG(vpin) OVER (PARTITION BY ticker ORDER BY bucket_id) AS delta_vpin
        FROM vpin_raw
    )"""
