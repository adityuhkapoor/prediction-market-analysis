"""Fetch Phase 1 insider case trades from Polymarket Data API.

Saves normalized trades (ticker, count, taker_side, yes_price, created_time)
directly to data/case_study/phase1/{case_id}_normalized_trades.parquet.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analysis.util.insider_cases import all_cases

DATA_API_URL = "https://data-api.polymarket.com"
CACHE_DIR = Path("data/case_study/phase1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all_trades(condition_id: str, limit: int = 500) -> list[dict]:
    """Fetch all trades for a market from the Polymarket Data API.

    Uses offset pagination first, then falls back to timestamp cursor
    if the API's offset limit (~3500) is reached.
    """
    all_trades = []
    seen_keys = set()

    def _dedup_extend(batch):
        added = 0
        for t in batch:
            key = f"{t.get('transactionHash', '')}_{t.get('timestamp', '')}_{t.get('size', '')}_{t.get('outcome', '')}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_trades.append(t)
                added += 1
        return added

    with httpx.Client(timeout=30.0) as client:
        # Phase 1: offset-based pagination
        offset = 0
        hit_limit = False
        while True:
            resp = client.get(
                f"{DATA_API_URL}/trades",
                params={"market": condition_id, "limit": limit, "offset": offset},
            )
            if resp.status_code == 400:
                hit_limit = True
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            _dedup_extend(batch)
            print(f"  fetched {len(all_trades)} trades (offset={offset})")
            if len(batch) < limit:
                break
            offset += limit
            sleep(0.15)

        # Phase 2: if we hit the offset limit, page backward by timestamp
        if hit_limit and all_trades:
            print("  offset limit reached, switching to timestamp cursor...")
            oldest_ts = min(int(t.get("timestamp", 0)) for t in all_trades)
            while True:
                resp = client.get(
                    f"{DATA_API_URL}/trades",
                    params={"market": condition_id, "limit": limit, "before": oldest_ts},
                )
                if resp.status_code != 200:
                    break
                batch = resp.json()
                if not batch:
                    break
                added = _dedup_extend(batch)
                print(f"  fetched {len(all_trades)} trades (cursor before={oldest_ts})")
                if added == 0:
                    break
                new_oldest = min(int(t.get("timestamp", 0)) for t in batch)
                if new_oldest >= oldest_ts:
                    break
                oldest_ts = new_oldest
                sleep(0.15)

    return all_trades


def normalize_trades(trades: list[dict], case) -> pd.DataFrame:
    """Convert Data API trades to the VPIN-ready normalized schema."""
    records = []
    for t in trades:
        side = t["side"]  # BUY or SELL
        outcome = t.get("outcome", "")
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        ts = int(t.get("timestamp", 0))

        if size <= 0 or price <= 0:
            continue

        # Determine taker_side per CLAUDE.md convention:
        #   BUY YES → 'yes', SELL YES → 'no'
        #   BUY NO → 'no',  SELL NO → 'yes'
        if outcome.lower() == "yes":
            taker_side = "yes" if side == "BUY" else "no"
            yes_price = price * 100  # Convert 0-1 to cents
        elif outcome.lower() == "no":
            taker_side = "no" if side == "BUY" else "yes"
            yes_price = (1 - price) * 100
        else:
            # Fallback: use token ID matching
            asset = t.get("asset", "")
            if asset == case.yes_token_id:
                taker_side = "yes" if side == "BUY" else "no"
                yes_price = price * 100
            elif asset == case.no_token_id:
                taker_side = "no" if side == "BUY" else "yes"
                yes_price = (1 - price) * 100
            else:
                continue

        records.append({
            "ticker": case.condition_id,
            "count": size,
            "taker_side": taker_side,
            "yes_price": yes_price,
            "created_time": datetime.fromtimestamp(ts, tz=timezone.utc),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("created_time").reset_index(drop=True)
    return df


def main():
    cases = all_cases()
    print(f"Fetching trades for {len(cases)} Phase 1 cases\n")

    for case in cases:
        print(f"=== {case.case_id} ===")
        print(f"  market: {case.market_slug}")
        print(f"  condition_id: {case.condition_id}")

        cache_path = CACHE_DIR / f"{case.case_id}_normalized_trades.parquet"

        if not case.condition_id:
            print("  SKIP — no condition_id\n")
            continue

        raw_trades = fetch_all_trades(case.condition_id)
        print(f"  total raw trades: {len(raw_trades)}")

        if not raw_trades:
            print("  SKIP — no trades returned\n")
            continue

        df = normalize_trades(raw_trades, case)
        print(f"  normalized trades: {len(df)}")

        if len(df) == 0:
            print("  SKIP — normalization produced 0 rows\n")
            continue

        # Quality checks
        ts = df["created_time"]
        print(f"  date range: {ts.min()} → {ts.max()}")
        print(f"  total volume: {df['count'].sum():,.0f} tokens")
        print(f"  yes_price range: [{df['yes_price'].min():.1f}, {df['yes_price'].max():.1f}]")
        print(f"  taker_side split: {df['taker_side'].value_counts().to_dict()}")

        df.to_parquet(cache_path, index=False)
        print(f"  saved to {cache_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
