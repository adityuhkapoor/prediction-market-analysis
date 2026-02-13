# VPIN Insider Trading Detection — Two-Phase Cross-Platform Study

## Status: Phase 1 — Code Complete, Awaiting Data

---

## Goal

Test whether VPIN can detect insider trading in prediction markets.

- **Phase 1 (Polymarket):** Validate against 5 confirmed insider cases with criminal/investigative documentation
- **Phase 2 (Kalshi):** Apply validated detection rule to scan hundreds of Kalshi events

## Pre-Registered Hypotheses

| ID | Hypothesis | Test | Min effect |
|----|-----------|------|------------|
| H1 | VPIN stochastically larger during insider windows | KS (alt='less') + MW + Permutation 10K | D >= 0.15 |
| H2 | Signed flow predicts price at spikes, insider only | Binomial vs 0.5 | HR >= 60%, h >= 0.20 |
| H3 | VPIN leads price volatility when insiders active | Cross-correlogram + bootstrap CI 1K | Peak lag >= +3 |
| H4 | |ΔVPIN| spikes sharply at insider entry | KS on |ΔVPIN| insider vs non | 2x max ratio |

FDR: BH q=0.10 (Phase 1), q=0.05 (Phase 2).

## Decision Criteria

- **Strong positive (→ Phase 2):** >= 3/5 cases show H1 sig after FDR + D >= 0.15 + HR >= 60%
- **Weak positive (→ cautious Phase 2):** 2/5 cases
- **Negative (study ends):** < 2 cases — report honestly

---

## Confirmed Insider Cases (Phase 1)

| Case ID | Market | Profit | Evidence | Status |
|---------|--------|--------|----------|--------|
| `ricosuave666` | Israel/Iran strikes | $152K | DOJ criminal charges | Need condition_id, wallet |
| `0xafee_google` | Google Year in Search 2025 | ~$1M | 22/23 correct | Need condition_id, wallet |
| `nobel_2025` | Nobel Peace Prize 2025 | ~$90K | Norway investigating | Need condition_id |
| `sb_halftime` | Super Bowl LX halftime | ~$17K | 17/20 correct, day-old account | Need condition_id |
| `maduro_capture` | Maduro capture | $400K+ | Hours before classified raid | Need condition_id |

---

## Completion Status

### Step 1: Normalization Layer — DONE
- [x] `normalization.py` with `kalshi_normalize_cte()` and `polymarket_normalize_cte()`
- [x] Kalshi: trivial column rename
- [x] Polymarket: joins trades+blocks+markets, derives side from (maker_asset_id, token mapping)
- [x] Price conversion: decimal (0-1) → cents (0-100)
- [x] `test_normalization.py`: 9 tests pass

### Step 2: Case Registry — DONE (partial)
- [x] `insider_cases.py` with `InsiderCase` dataclass + 5 cases
- [ ] **Polymarket condition_ids** — need to research per case
- [ ] **CLOB token IDs** (yes_token_id, no_token_id) — need Gamma API lookup
- [ ] **Wallet addresses** — need Polygonscan/Dune/DOJ complaint research
- [ ] Control market IDs — deferred until primary cases populated

### Step 3: Statistical Tests — DONE
- [x] `stats.py`: ks_test, mann_whitney_test, permutation_test, binomial_test, bootstrap_peak_lag, apply_bh_fdr
- [x] `test_stats.py`: 17 tests pass (all verified against scipy)

### Step 4: ΔVPIN — DONE
- [x] `vpin.py` modified: `vpin_raw` → `vpin_series` wrapper CTE with `delta_vpin`
- [x] Backward compatible — all 6 existing VPIN tests still pass
- [x] NULL for first bucket per market

### Step 5: Phase 1 Data Acquisition — BLOCKED
- [ ] Polymarket trades fetched for 5 case markets
- [ ] Block timestamps joined
- [ ] Normalization applied
- **Blocker:** Need condition_ids + token IDs populated in insider_cases.py
- **Blocker:** Need Polygon RPC access or pre-fetched parquet files

### Step 6: Phase 1 Experiments — CODE COMPLETE, NOT RUN
- [x] Notebook coded: `phase1_validation.ipynb` (10 cells)
  - Cell 1: Pre-registration header
  - Cell 2: Data acquisition with quality checks
  - Cell 3: VPIN computation with insider window marking
  - Cell 4: H1 — temporal comparison (KS + MW + permutation)
  - Cell 5: H4 — ΔVPIN regime detection
  - Cell 6: H2 — directional accuracy (binomial test)
  - Cell 7: H3 — lead-lag (cross-correlogram + bootstrap)
  - Cell 8: Permutation null (1000 shuffles)
  - Cell 9: FDR correction + per-case summary + verdict
  - Cell 10: Decision rule extraction → phase1_report.md
- [ ] **Not yet run** — waiting on data

### Step 7: Sensitivity Analysis — CODE COMPLETE, NOT RUN
- [x] `phase1_sensitivity.ipynb` coded (6 cells)
- [x] Grid: 4 bucket_sizes × 4 lookbacks × 3 windows = 48 configs per case
- [ ] **Not yet run**

### Step 8: Phase 1 Freeze — PENDING
- [ ] `phase1_report.md` written with findings + decision rule
- [ ] Committed to git before Phase 2

### Step 9: Phase 2 Kalshi Scan — CODE COMPLETE, NOT RUN
- [x] `phase2_kalshi_scan.ipynb` coded (7 cells)
- [ ] **Not yet run** — depends on Phase 1 result

---

## Next Steps (in order)

1. **Research Polymarket market IDs.** For each case, find:
   - Polymarket slug / condition_id
   - YES and NO CLOB token IDs
   - Known insider wallet addresses (from DOJ complaints, news articles, Dune queries)
   - Sources: Gamma API (`GET /markets?slug=...`), Polygonscan, Dune Analytics, news articles

2. **Populate `insider_cases.py`** with researched values.

3. **Fetch Polymarket on-chain data.** Either:
   - Run the blockchain indexer: `uv run main.py index` (needs POLYGON_RPC)
   - Or manually place pre-fetched parquet files in `data/polymarket/`

4. **Run Phase 1 notebook** end-to-end.

5. **Run sensitivity notebook.**

6. **Commit phase1_report.md** and freeze.

7. **Run Phase 2** (if Phase 1 positive).

---

## Key Files to Read When Resuming

| Priority | File | Why |
|----------|------|-----|
| 1 | `CLAUDE.md` (repo root) | Full project context |
| 2 | This file (`docs/PLAN.md`) | Study plan + progress |
| 3 | `src/analysis/util/vpin.py` | Core VPIN engine |
| 4 | `src/analysis/util/normalization.py` | Platform normalization CTEs |
| 5 | `src/analysis/util/insider_cases.py` | Case registry (needs population) |
| 6 | `src/analysis/util/stats.py` | Statistical test functions |
| 7 | `notebooks/phase1_validation.ipynb` | Phase 1 notebook (ready to run) |
| 8 | `src/indexers/polymarket/blockchain.py` | How Polymarket trades are fetched |
| 9 | `docs/DEVLOG.md` | History of all changes + rationale |

---

## Open Questions

- **Polymarket data availability:** Is `data/polymarket/` populated? If not, need POLYGON_RPC.
- **Wallet addresses:** DOJ complaint for ricosuave666 should have the Polygon address. 0xafEe widely reported.
- **Token ID mapping:** Gamma API `GET /markets?slug=<slug>` returns `clobTokenIds` JSON array.
- **VPIN bucket unit for Polymarket:** Token count or USDC volume? Whale insiders (one $400K trade) vs retail insiders (fifty $600 trades) produce different profiles.
- **Wash trading contamination:** Chaos Labs found ~1/3 of presidential election volume was wash trading. May affect our target markets.

## Assumptions

1. Polymarket on-chain data reflects real trading (may not hold — wash trading risk)
2. Block timestamp interpolation (~200s error) acceptable for VPIN on volume buckets
3. `maker_asset_id = '0'` reliably indicates USDC (true for CTF Exchange contract)
4. Binary markets only (exclude multi-outcome)
5. VPIN is meaningful for prediction markets (exactly what Phase 1 tests)
6. Exact taker-side classification eliminates Lee-Ready noise (genuine advantage over equities)
7. Volume-synchronized bucketing appropriate (insiders generate volume when they trade)

---

## Test Suite

```
32 tests pass:
  test_vpin_cte.py         — 6 tests (VPIN math)
  test_normalization.py    — 9 tests (platform normalization + delta_vpin)
  test_stats.py            — 17 tests (all stat functions vs scipy)

2 tests skip (pre-existing):
  test_vpin_signal.py      — missing tenacity module
  test_mispricing.py       — missing tenacity module
```

Run: `uv run pytest tests/test_vpin_cte.py tests/test_normalization.py tests/test_stats.py -v`
