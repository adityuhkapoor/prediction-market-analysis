# CLAUDE.md — Project Context for Autonomous Agents

## What This Project Is

A quantitative study testing whether VPIN (Volume-Synchronized Probability of Informed Trading) can detect insider trading in prediction markets (Polymarket, Kalshi). Two-phase design: validate against known insiders, then scan at scale.

## How to Run Things

```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests (32 pass, 2 skip due to missing tenacity)
uv run main.py analyze           # Run all Kalshi analyses
uv run main.py index             # Run all indexers
make lint                        # Ruff check
make format                      # Ruff auto-fix
```

## Architecture

```
src/
  analysis/
    util/
      vpin.py              # Core: vpin_cte() and qualified_trades_cte() — DuckDB SQL generators
      normalization.py     # Platform-specific CTEs producing common schema for vpin_cte()
      insider_cases.py     # InsiderCase dataclass + 5-case registry (Phase 1 targets)
      stats.py             # Statistical tests: KS, MW, permutation, binomial, bootstrap, BH-FDR
      categories.py        # 570+ pattern rules mapping event_tickers to categories
    kalshi/
      vpin_signal.py       # VPINSignalAnalysis — tests VPIN → volatility/direction/resolution
      vpin_categories.py   # VPIN decomposition by market category
    polymarket/            # Polymarket-specific analyses (volume, win rate)
  indexers/
    kalshi/
      client.py            # KalshiClient with pagination + retry
      models.py            # Trade and Market dataclasses
    polymarket/
      blockchain.py        # PolygonClient — fetches OrderFilled events from Polygon chain
      blocks.py            # Block number → timestamp mapping (sampled every 100 blocks)
      client.py            # PolymarketClient — Gamma API + Data API
      models.py            # Market and Trade dataclasses
  common/
    analysis.py            # Analysis base class with run(), save(), progress()

notebooks/
  phase1_validation.ipynb  # Phase 1: validate VPIN against 5 Polymarket insider cases
  phase1_sensitivity.ipynb # Robustness: 48-config parameter grid
  phase2_kalshi_scan.ipynb # Phase 2: scan Kalshi markets with frozen decision rule
  vpin_insider_trading.ipynb  # Exploratory: Super Bowl halftime (21 Kalshi markets)
  vpin_analysis.ipynb      # Pedagogy: VPIN introduction + 4 tests

tests/
  conftest.py              # DuckDB connection + synthetic_trades factory
  test_vpin_cte.py         # 6 tests: VPIN math on synthetic data
  test_normalization.py    # 9 tests: Kalshi passthrough, Polymarket side derivation
  test_stats.py            # 17 tests: all stat functions against scipy/analytical values
```

## VPIN Pipeline (how data flows)

```
1. qualified_trades_cte(trades_dir, markets_dir)
   → CTE: market_info, qualified_markets, trades

2. vpin_cte("trades", bucket_size=200, lookback=10)
   → CTE: volume_running, volume_buckets, bucket_stats, vpin_raw, vpin_series
   → Output columns: ticker, bucket_id, v_yes, v_no, total_vol, avg_price,
     bucket_start, bucket_end, order_imbalance, vpin, signed_flow,
     window_size, delta_vpin

3. vpin_cte expects 5 input columns:
   ticker, count, taker_side, yes_price, created_time
   → normalization.py produces these from platform-specific raw data
```

## Polymarket Trade Side Derivation

```
maker_asset_id = '0'  → maker provides USDC → taker BUYS outcome tokens
maker_asset_id != '0' → maker provides tokens → taker SELLS outcome tokens

Mapping to taker_side (what vpin_cte expects):
  BUY YES tokens  → taker_side = 'yes'
  SELL YES tokens → taker_side = 'no'
  BUY NO tokens   → taker_side = 'no'
  SELL NO tokens  → taker_side = 'yes'
```

## KS Test Convention (Important)

scipy `ks_2samp(a, b, alternative='less')` tests that `a` is stochastically LARGER than `b`. This is counterintuitive — "less" refers to F_a(x) < F_b(x), meaning `a`'s CDF is below `b`'s, meaning `a` has more mass at higher values.

When testing "insider VPIN > non-insider VPIN", use `alternative='less'`.

## Current State

- **Branch:** `vpin-analysis` (ahead of main)
- **All new code written.** normalization.py, insider_cases.py, stats.py, delta_vpin in vpin.py, 3 notebooks
- **32 tests pass.** 2 pre-existing tests skip (missing tenacity module — not our code)
- **Lint clean.** All new files pass ruff

## What Remains (see docs/PLAN.md for full detail)

1. **Populate Polymarket market IDs** in insider_cases.py — condition_ids, token IDs, wallet addresses
2. **Fetch Polymarket on-chain data** for the 5 insider case markets
3. **Run Phase 1 notebook** — the experiments are coded, just need data
4. **Run sensitivity analysis** — parameter grid over bucket_size × lookback × window
5. **Commit phase1_report.md** — freeze findings before Phase 2
6. **Run Phase 2 notebook** — Kalshi scan (only if Phase 1 positive)

## Conventions

- All randomness uses `np.random.default_rng(seed=42)`
- DuckDB for all SQL — no pandas SQL or sqlite
- Parquet for all data caching
- BH-FDR for multiple testing (Phase 1: q=0.10, Phase 2: q=0.05)
- Tests alongside code, not batched at the end
- Delete unused code completely — no commented-out blocks

## Environment Variables

- `ANTHROPIC_API_KEY` — for Claude Code
- `POLYGON_RPC` — Polygon RPC endpoint for Polymarket blockchain indexer
- `POLYMARKET_START_BLOCK` — defaults to 33605403 (CTF Exchange deployment)

## Data Layout (gitignored)

```
data/
  kalshi/
    trades/*.parquet       # Kalshi trade history (36GB full dataset)
    markets/*.parquet      # Kalshi market metadata
  polymarket/
    trades/*.parquet       # Polymarket OrderFilled events from Polygon
    blocks/*.parquet       # Block number → timestamp mapping
    markets/*.parquet      # Polymarket market metadata from Gamma API
  case_study/
    *_trades.parquet       # Cached trades for specific case studies
    phase1/                # Normalized trades for Phase 1 insider cases
```
