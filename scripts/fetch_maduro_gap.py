"""Fetch missing maduro trades from blockchain (Jan 2 18:00 - Jan 3 07:15 UTC).

Uses 10-block chunks to stay within Alchemy free tier limits.
Saves normalized trades and merges with existing Data API trades.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.indexers.polymarket.blockchain import PolygonClient, CTF_EXCHANGE, NEGRISK_CTF_EXCHANGE

YES_TOKEN = int("24918067747661759048720135607687934209172945708698786072564741153847301974566")
NO_TOKEN = int("98590363695810976489755414330258191966783137178752226225745237702303580425394")
CONDITION_ID = "0x580adc1327de9bf7c179ef5aaffa3377bb5cb252b7d6390b027172d43fd6f993"

START_BLOCK = 81128389  # ~Jan 2 18:00 UTC
END_BLOCK = 81152689    # ~Jan 3 07:15 UTC
CHUNK_SIZE = 10  # Alchemy free tier limit

CACHE_DIR = Path("data/case_study/phase1")


def fetch_gap_trades():
    client = PolygonClient()
    maduro_trades = []
    errors = 0
    total_chunks = (END_BLOCK - START_BLOCK) // CHUNK_SIZE + 1

    print(f"Scanning blocks {START_BLOCK} - {END_BLOCK} ({total_chunks} chunks of {CHUNK_SIZE})")
    print(f"Trying both CTF and NegRisk exchanges\n")

    for exchange_name, exchange_addr in [("CTF", CTF_EXCHANGE), ("NegRisk", NEGRISK_CTF_EXCHANGE)]:
        print(f"\n--- {exchange_name} Exchange ---")
        exchange_trades = []

        for i, block_start in enumerate(range(START_BLOCK, END_BLOCK, CHUNK_SIZE)):
            block_end = min(block_start + CHUNK_SIZE - 1, END_BLOCK)

            try:
                trades = client.get_trades(block_start, block_end, exchange_addr)
                for t in trades:
                    token_traded = t.taker_asset_id if t.is_buy else t.maker_asset_id
                    if token_traded == YES_TOKEN or token_traded == NO_TOKEN:
                        exchange_trades.append(t)
            except Exception as e:
                errors += 1
                if errors % 50 == 1:
                    print(f"  Error at block {block_start}: {e}")

            if (i + 1) % 200 == 0:
                pct = (i + 1) / total_chunks * 100
                print(f"  [{pct:.0f}%] scanned {i+1}/{total_chunks} chunks, "
                      f"found {len(exchange_trades)} trades, {errors} errors")

            # Rate limiting: ~5 req/sec for free tier
            time.sleep(0.2)

        print(f"  {exchange_name}: {len(exchange_trades)} maduro trades found")
        maduro_trades.extend(exchange_trades)

    print(f"\nTotal gap trades found: {len(maduro_trades)}")

    if not maduro_trades:
        print("No gap trades found. The existing Data API data will be used as-is.")
        return

    # Get timestamps for the trades
    print("Fetching block timestamps...")
    block_cache = {}
    for t in maduro_trades:
        if t.block_number not in block_cache:
            try:
                block_cache[t.block_number] = client.get_block_timestamp(t.block_number)
                time.sleep(0.1)
            except Exception:
                block_cache[t.block_number] = 0

    # Normalize to VPIN schema
    records = []
    for t in maduro_trades:
        ts = block_cache.get(t.block_number, 0)
        if ts == 0:
            continue

        token_traded = t.taker_asset_id if t.is_buy else t.maker_asset_id
        is_yes_token = token_traded == YES_TOKEN

        # Taker side derivation
        if t.is_buy and is_yes_token:
            taker_side = "yes"
        elif t.is_buy and not is_yes_token:
            taker_side = "no"
        elif not t.is_buy and is_yes_token:
            taker_side = "no"
        else:
            taker_side = "yes"

        price = t.price
        if not is_yes_token:
            price = 1.0 - price

        records.append({
            "ticker": CONDITION_ID,
            "count": t.size,
            "taker_side": taker_side,
            "yes_price": price * 100,
            "created_time": datetime.fromtimestamp(ts, tz=timezone.utc),
        })

    gap_df = pd.DataFrame(records)
    print(f"Normalized gap trades: {len(gap_df)}")

    if len(gap_df) == 0:
        return

    # Merge with existing Data API trades
    existing_path = CACHE_DIR / "maduro_capture_normalized_trades.parquet"
    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        merged = pd.concat([gap_df, existing_df], ignore_index=True)
        merged = merged.sort_values("created_time").reset_index(drop=True)
        # Backup original
        existing_path.rename(existing_path.with_suffix(".parquet.bak"))
    else:
        merged = gap_df.sort_values("created_time").reset_index(drop=True)

    merged.to_parquet(existing_path, index=False)
    print(f"\nMerged and saved {len(merged)} total trades to {existing_path}")
    print(f"Date range: {merged['created_time'].min()} → {merged['created_time'].max()}")


if __name__ == "__main__":
    fetch_gap_trades()
