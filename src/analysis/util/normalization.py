"""Platform-specific normalization CTEs for VPIN computation.

Each function produces a DuckDB CTE named ``trades`` with the common schema
expected by ``vpin_cte()``:

    ticker       TEXT       — market identifier
    count        DOUBLE     — trade quantity (token units)
    taker_side   TEXT       — 'yes' or 'no'
    yes_price    DOUBLE     — price in cents (0-100)
    created_time TIMESTAMP  — trade timestamp
"""


def kalshi_normalize_cte(
    trades_source: str,
    ticker_column: str = "ticker",
) -> str:
    """Normalize Kalshi trade data into the common VPIN schema.

    Kalshi trades already have the correct columns (ticker, count, taker_side,
    yes_price, created_time), so this is a trivial passthrough / rename.

    Args:
        trades_source: DuckDB table name or parquet glob containing raw Kalshi trades.
        ticker_column: Column name used as the market identifier.

    Returns:
        A DuckDB CTE string defining ``trades`` with the common schema.
    """
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
    """Normalize Polymarket on-chain trade data into the common VPIN schema.

    Polymarket OrderFilled events have a different structure:
    - ``maker_asset_id``, ``taker_asset_id``, ``maker_amount``, ``taker_amount``
    - No ``taker_side`` column — must be derived from asset IDs + token mapping

    Side derivation logic:
        maker_asset_id = '0'  → maker provides USDC, taker buys outcome tokens
        maker_asset_id != '0' → maker provides tokens, taker sells outcome tokens

        The token being traded determines whether it is a YES or NO token.
        We join against the markets table to map token IDs to outcomes.

    Price calculation:
        When taker buys:  price = maker_amount / taker_amount  (USDC per token)
        When taker sells: price = taker_amount / maker_amount  (USDC per token)
        Scaled to cents (×100).

    Quantity:
        Token amount / 1e6  (both USDC and outcome tokens use 6 decimals).

    Args:
        trades_source: DuckDB table/CTE with Polymarket blockchain trades.
            Expected columns: block_number, maker_asset_id, taker_asset_id,
            maker_amount, taker_amount, transaction_hash, log_index.
        blocks_source: DuckDB table/CTE with block_number → timestamp mapping.
        markets_source: DuckDB table/CTE with Polymarket market metadata.
            Expected columns: condition_id, clob_token_ids (JSON array of
            [yes_token_id, no_token_id]).

    Returns:
        A DuckDB CTE string defining ``trades`` with the common schema.
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
