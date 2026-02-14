"""Platform-specific normalization into the common VPIN schema.

Output columns: ticker, count, taker_side, yes_price, created_time
"""


def kalshi_normalize_cte(
    trades_source: str,
    ticker_column: str = "ticker",
) -> str:
    """Kalshi trades already match the schema — trivial passthrough."""
    return f"""trades AS (
        SELECT
            {ticker_column} AS ticker,
            count,
            taker_side,
            yes_price,
            created_time
        FROM {trades_source}
    )"""


def polymarket_normalize_cte(
    trades_source: str,
    blocks_source: str,
    markets_source: str,
) -> str:
    """Normalize Polymarket on-chain OrderFilled events.

    Side derivation:
        maker_asset_id = '0'  → taker buys tokens (BUY)
        maker_asset_id != '0' → taker sells tokens (SELL)
        BUY YES → 'yes', SELL YES → 'no', BUY NO → 'no', SELL NO → 'yes'

    Price: USDC/token ratio scaled to cents (×100).
    Quantity: token amount / 1e6 (both USDC and tokens use 6 decimals).
    """
    return f"""polymarket_with_ts AS (
        SELECT
            t.*,
            b.timestamp AS created_time
        FROM {trades_source} t
        INNER JOIN {blocks_source} b ON t.block_number = b.block_number
    ),
    polymarket_with_side AS (
        SELECT
            pw.*,
            CASE
                WHEN pw.maker_asset_id = '0' THEN 'BUY'
                ELSE 'SELL'
            END AS trade_side,
            CASE
                WHEN pw.maker_asset_id = '0' THEN pw.taker_asset_id
                ELSE pw.maker_asset_id
            END AS traded_token_id
        FROM polymarket_with_ts pw
    ),
    trades AS (
        SELECT
            m.condition_id AS ticker,
            CASE
                WHEN ps.maker_asset_id = '0'
                    THEN ps.taker_amount / 1e6
                ELSE ps.maker_amount / 1e6
            END AS count,
            CASE
                WHEN ps.trade_side = 'BUY' AND ps.traded_token_id = m.yes_token_id THEN 'yes'
                WHEN ps.trade_side = 'BUY' AND ps.traded_token_id = m.no_token_id THEN 'no'
                WHEN ps.trade_side = 'SELL' AND ps.traded_token_id = m.yes_token_id THEN 'no'
                WHEN ps.trade_side = 'SELL' AND ps.traded_token_id = m.no_token_id THEN 'yes'
            END AS taker_side,
            CASE
                WHEN ps.maker_asset_id = '0'
                    THEN (ps.maker_amount::DOUBLE / ps.taker_amount) * 100
                ELSE (ps.taker_amount::DOUBLE / ps.maker_amount) * 100
            END AS yes_price,
            ps.created_time
        FROM polymarket_with_side ps
        INNER JOIN {markets_source} m
            ON ps.traded_token_id = m.yes_token_id
            OR ps.traded_token_id = m.no_token_id
        WHERE ps.taker_amount > 0 AND ps.maker_amount > 0
    )"""
