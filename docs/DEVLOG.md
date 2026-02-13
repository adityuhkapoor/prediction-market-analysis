# Devlog

## 2026-02-12: Add test suite (22 tests)

**New files:** `tests/conftest.py`, `tests/test_vpin_cte.py`, `tests/test_categories.py`, `tests/test_vpin_signal.py`, `tests/test_mispricing.py`

### What's covered

- **VPIN math (6 tests):** balanced→low VPIN, imbalanced→VPIN=1.0, signed flow polarity, window ramp, bucket assignment
- **Categories (8 tests):** known tickers (Sports, Politics, Crypto), unknown/None/empty→"Other", hierarchy 3-tuple, GROUP_COLORS coverage
- **Signal analysis (4 tests):** first-half truncation, MIN_MARKETS guard, FDR key presence, FDR values >= raw
- **Mispricing model (4 tests):** log_volume formula, temporal split monotonicity, shared bin edges, extrapolated price→zero adj

### Also fixed

- `get_hierarchy(None)` crashed with `AttributeError`. Added null guard to return `("Other", "Other", "")`.
- Added `[tool.pytest.ini_options] pythonpath = ["."]` to `pyproject.toml` so pytest can find `src.*` imports.

### Notes for future sessions

All tests use synthetic data — no real data files needed. Run with `uv run pytest tests/ -v`.

---

## 2026-02-12: Notebook imports from src/ instead of inline code

**File:** `notebooks/vpin_analysis.ipynb`

### Problem

The notebook duplicated `vpin_cte()`, `MIN_TRADES`, `get_group()`, `GROUP_COLORS`, and the full CTE chain from `src/`. It also had two bugs that `src/` already fixed:
1. `log_volume` used `groupby().transform('sum')` on lifetime total volume (data leakage — Step 2 bug)
2. `pd.cut` computed separate bin edges for train and test sets (bin-edge leakage — Step 3 bug)

### Fix

1. Added setup cell with `sys.path.insert(0, _repo_root)` — hardened with validation that `src/analysis/util/vpin.py` exists before trusting the path.
2. Replaced inline `vpin_cte` and `MIN_TRADES` with `from src.analysis.util.vpin import vpin_cte, MIN_TRADES, qualified_trades_cte`.
3. Replaced inline CTE chain with `qualified_trades_cte(...)`.
4. Replaced inline `CATEGORY_PATTERNS`, `get_group`, and `GROUP_COLORS` with `from src.analysis.util.categories import get_group, GROUP_COLORS`. Note: `src/` has 568 patterns vs notebook's 60, so some "Other" tickers now get proper group assignments.
5. Fixed `log_volume`: `np.log1p((mid_df['bucket_id'] + 1) * BUCKET_SIZE)`.
6. Fixed `pd.cut`: `retbins=True` on training call, passed `bin_edges` to test call.

### Notes for future sessions

Do NOT add `random_state=42` to the `LogisticRegression(max_iter=1000)` calls — `lbfgs` is deterministic. The notebook cell-1 is Colab-specific (downloads 33GB dataset) and cannot be executed locally.

---

## 2026-02-12: Add data validation and error handling

**Files:** `src/common/analysis.py`, `src/analysis/kalshi/vpin_signal.py`, `src/analysis/kalshi/mispricing_model.py`, `src/analysis/kalshi/vpin_categories.py`, `src/indexers/kalshi/trades.py`

### Problem

1. Analyses could silently produce empty DataFrames without any error — downstream consumers (CSV save, figure generation) would get cryptic failures.
2. `mispricing_model.py` had a `if vpin_df.empty: return pd.DataFrame()` guard that silently returned empty output instead of failing loudly.
3. `trades.py` had `except Exception: pass` that swallowed all errors during dedup loading.

### Fix

Two-layer validation (defense in depth):

1. **Safety net in `save()`:** After `self.run()`, raises `ValueError` if `output.data` is an empty DataFrame. Catches all 26 Analysis subclasses universally. `data=None` (legitimate — no tabular output) is fine; `data=pd.DataFrame()` with 0 rows is almost certainly a bug.

2. **`_require_data()` method on `Analysis`:** Called after major DuckDB queries in each VPIN analysis. Provides precise error messages with context (which query failed, how many rows returned, minimum expected).

3. **Removed silent empty guard** from `mispricing_model.py:_extract_features` — the `save()` safety net now catches this.

4. **Fixed `except Exception: pass`** in `trades.py` — now logs a warning with the exception message.

### Notes for future sessions

When adding new Analysis subclasses, call `self._require_data()` after major DuckDB queries for precise error messages. The `save()` safety net catches empty output regardless, but `_require_data` gives better diagnostics.

---

## 2026-02-12: Extract shared market filtering CTE

**Files:** `src/analysis/util/vpin.py`, `src/analysis/kalshi/vpin_signal.py`, `src/analysis/kalshi/mispricing_model.py`, `src/analysis/kalshi/vpin_categories.py`

### Problem

Three files duplicated the same `market_info → qualified_markets → trades` CTE chain (~15 lines each). Divergence risk: any fix to market filtering logic would need to be applied in 3 places.

### Fix

Added `qualified_trades_cte(trades_dir, markets_dir)` to `src/analysis/util/vpin.py`. All three consumer files now call this shared function. Also removed the redundant `INNER JOIN trade_counts tc` from `mispricing_model.py`'s final SELECT — structurally redundant since `trades` already filters through `qualified_markets`.

### Design decisions

- **Superset columns in market_info:** Always selects `ticker, event_ticker, result, close_time` even when a consumer doesn't need all of them. DuckDB inlines CTEs and the optimizer removes unused columns, so no performance cost.
- **trade_counts JOIN removal:** The original `mispricing_model.py` joined `vpin_series` with `trade_counts` in the final SELECT. Since `vpin_series` is built from `trades`, which already filters through `qualified_markets`, every ticker in `vpin_series` is guaranteed to be in `qualified_markets`. The join was structurally redundant.

### Verification

Output diffs: vpin_signal and vpin_categories show only floating-point noise (15th+ decimal place). mispricing_model shows small variations from sort-stability in temporal split (pre-existing, not introduced by refactor). Market counts identical (60 train, 26 test).

---

## 2026-02-12: Add Benjamini-Hochberg FDR correction to regression p-values

**File:** `src/analysis/kalshi/vpin_signal.py`

### Problem

`vpin_signal.py` runs 6 regression tests (3 volatility horizons k=1,5,10, 3 direction horizons) and reports raw p-values. Without multiple comparison correction, the family-wise error rate is 1-(1-0.05)^6 ≈ 26%. Readers seeing six tests with p<0.05 may overestimate the evidence.

### Fix

Added `_apply_fdr_correction()` method using `scipy.stats.false_discovery_control(method="bh")`. Called after all regression tests in `run()`. Both raw and FDR-corrected p-values appear in CSV output. Figure stars now use FDR-corrected p-values.

### Design decisions

- **BH (not BY):** Our tests have positive regression dependency (different horizons on the same data), and BH is valid under PRDS (positive regression dependency on a subset). BY would be overly conservative here.
- **Stars reflect FDR:** Raw stars mislead readers — corrected stars reflect honest significance after accounting for multiple testing. Raw p-values remain in CSV for transparency.

### Results after correction

| Test | Raw p | FDR p | Significant? |
|------|-------|-------|-------------|
| vol_k1 | 0.009 | 0.019 | Yes * |
| vol_k5 | 0.001 | 0.003 | Yes ** |
| vol_k10 | 2.6e-7 | 1.5e-6 | Yes *** |
| dir_k1 | 0.063 | 0.063 | No |
| dir_k5 | 0.050 | 0.060 | No (flipped) |
| dir_k10 | 0.032 | 0.048 | Yes * |

All 3 volatility results survive. Direction k5 flips non-significant (FDR=0.060). Direction k10 barely survives.

### Notes for future sessions

If new regression tests are added (e.g., testing additional horizons or features), regenerate FDR corrections — they depend on the total number of tests in the family.

---

## 2025-02-12: Fix `pd.cut` bin-edge leakage in longshot adjustment

**File:** `src/analysis/kalshi/mispricing_model.py` (`_compute_longshot_adj`)

### Problem

`pd.cut(bins=20)` computes 20 evenly-spaced edges from the series' own `(min, max)`. The method called it independently on train and test sets, producing different bin edges. When test bin indices looked up `adj_map` (keyed by training bin indices), they got adjustments computed for a different price interval.

Example: train range [2, 97] → bin 0 covers [2, 6.75]. Test range [5, 95] → bin 0 covers [5, 9.5]. A test price of 8 falls in test-bin 0, but `adj_map[0]` holds the adjustment for train prices in [2, 6.75].

`longshot_adj` has a coefficient of ~2.4, making it the third-strongest feature in Model 2. Misaligned bin lookups injected noise into this feature for every test observation.

### Fix

Used `retbins=True` on the training call to capture bin edges, then passed those edges to the test call:

```python
train_df["price_bin"], bin_edges = pd.cut(train_df["price"], bins=20, labels=False, retbins=True)
# ...
test_df["price_bin"] = pd.cut(test_df["price"], bins=bin_edges, labels=False)
```

Test prices outside the training range get `NaN` bins, which `.map(adj_map).fillna(0)` already handles — no longshot adjustment for extrapolated prices.

### Also: removed dead `random_state=42` from `vpin_signal.py`

The original Step 3 plan was to add `random_state=42` to two `LogisticRegression` calls in `mispricing_model.py`. Investigation showed this is a no-op — `random_state` is ignored by `lbfgs` (the default and only solver used). The pipeline is already fully deterministic. Instead of adding misleading dead code, removed the one existing `random_state=42` from `vpin_signal.py` (line 215) that was introduced in Step 1.

---

## 2025-02-12: Fix `log_volume` data leakage in mispricing model

**File:** `src/analysis/kalshi/mispricing_model.py` (`_extract_features`)

### Problem

`log_volume` in Model 2 was computed from `total_contracts` — the sum of ALL trades across a market's entire lifetime. But the model observes features at the **midpoint** (~50% of volume). Using total lifetime volume tells the model how much trading will eventually happen, which is unknowable at prediction time. This is textbook data leakage.

### Fix

Replaced `log1p(total_contracts)` with `log1p((bucket_id + 1) * bucket_size)`, which approximates cumulative volume at the midpoint bucket using bucket geometry.

Since `bucket_id = FLOOR((cum_vol - 1) / bucket_size)`, cumulative volume at bucket k is in `[(k * bucket_size + 1), (k + 1) * bucket_size]`. The upper-bound approximation introduces at most one bucket of error, which is negligible after the log transform (e.g., `log1p(2401)=7.784` vs `log1p(2600)=7.864` at bucket 12).

Also removed `tc.total_contracts` from the SQL SELECT clause since it's no longer consumed as a feature. The `trade_counts` CTE and its JOIN remain (needed for the HAVING >= 500 filter).

### What didn't change

- Models 0 and 1 are unaffected (they don't use `log_volume`)
- No SQL CTE changes — the fix is purely in Python feature derivation
- The `tc` JOIN on the final SELECT is technically redundant (`trades` already filters through `trade_counts`) but left for a future cleanup pass

---

## 2025-02-12: Fix forward-looking feature contamination in resolution prediction

**File:** `src/analysis/kalshi/vpin_signal.py` (`_test_resolution_prediction`)

### Problem

After fixing in-sample evaluation (previous entry), 5-fold cross-validated resolution prediction reported **97% Brier skill** — worse than the original 69% in terms of honesty. The cross-validation was working correctly (each fold genuinely held out), but the features themselves encoded the outcome:

- `mean_signed_flow` averaged over the entire market lifetime cleanly separated YES (+0.33 avg) and NO (-0.36 avg) markets
- Late-market flow reflects outcome certainty as participants converge — this isn't prediction, it's tautology
- The question "does cumulative trading direction correlate with which side won?" trivially answers itself

For the thesis "informed flow predicts resolution", features must be strictly causal — only data available at the prediction point.

### Fix

Truncated features to the **first 50% of each market's volume buckets**:

```python
bucket_counts = vpin_df.groupby("ticker")["bucket_id"].transform("count")
bucket_rank = vpin_df.groupby("ticker")["bucket_id"].rank(method="first")
early_df = vpin_df[bucket_rank <= (bucket_counts / 2)].copy()
```

Also replaced `last_price` baseline (final bucket price — forward-looking relative to the cutoff) with `cutoff_price` (price at the last bucket in the early half).

### Rationale

- Tests 1 (volatility) and 2 (direction) already use `shift(-k)` — genuinely forward-looking predictions
- Test 3 should match: features from time T predict outcome after T
- 50% cutoff is simple, one parameter, no special tuning
- If skill survives truncation, that's a novel finding; if it drops to zero, that's honest

### Results

| Metric | Lifetime (before) | First-half (after) |
|--------|-------------------|-------------------|
| n_markets | 86 | 85 |
| baseline_brier_mean | 0.2107 | 0.2138 |
| baseline_brier (last/cutoff) | 0.2083 | 0.2126 |
| model_brier | 0.0049 | 0.0048 |
| skill vs mean | 97.7% | 97.8% |
| skill vs cutoff | 97.6% | 97.7% |
| coef_signed_flow | 9.37 | 9.34 |

**Skill survived truncation.** The signed flow gap between YES and NO markets is essentially identical in the first half (0.688) vs full lifetime (0.686).

### Open concern: 97% skill is still suspect

97% Brier skill from logistic regression on 85 markets is extraordinary. The first-half truncation rules out late-market certainty convergence, but other explanations remain:

- **Mechanical correlation**: if `taker_side` maps asymmetrically to YES/NO outcomes (e.g., YES markets attract more YES takers by construction), `signed_flow` separates outcomes trivially without any informed trading
- **Event-level leakage**: markets from the same `event_ticker` can land in both train and test CV folds — correlated markets inflate scores
- **Price doing the heavy lifting**: `mean_price` in the first half already encodes market consensus; the model gets this for free alongside signed flow

### TODO: harden before trusting

1. **Permutation test** — shuffle `result` labels, rerun CV, check skill drops to ~0. If it doesn't, signal is mechanical.
2. **`GroupKFold(groups=event_ticker)`** — replace 5-fold CV to eliminate event leakage.
3. **Ablation** — run model with `signed_flow` excluded. If `mean_price` + `mean_vpin` alone show similar skill, signed flow isn't the driver.

---

## 2025-02-12: Fix in-sample evaluation in resolution prediction

**File:** `src/analysis/kalshi/vpin_signal.py` (`_test_resolution_prediction`)

### Problems

**1. Arbitrary midpoint sampling.**
The old code picked a single VPIN observation at the 50th percentile volume bucket per market. No literature supports this choice — VPIN is designed as a continuous rolling metric (ELP2012, Wu et al., Low et al.), not sampled at a single arbitrary point. The midpoint is an undocumented free parameter that risks data snooping (Andersen-Bondarenko critique).

**2. In-sample evaluation.**
```python
model.fit(features, target)
model_probs = model.predict_proba(features)  # same data used for train and test
```
The model saw the answers during training, then was evaluated on the same answers. This inflated the reported 69% Brier improvement. With 3 features and only 86 observations, in-sample logistic regression will always look good.

**3. Single weak baseline.**
Only compared against mean market price as a probability estimate. No comparison against last traded price (a much harder baseline — the market's most recent and most informed assessment).

**4. No feature scaling.**
Raw features fed into LogisticRegression without standardization. Coefficients not comparable across features with different scales.

**5. Inline imports / dead parameters.**
`sklearn` imported inside the method body. `con` parameter passed but never used.

### Changes

**Aggregation:** Replaced midpoint sampling with full time-series aggregation per market — `mean(vpin)`, `mean(signed_flow)`, `mean(avg_price)`. Also extract `last_price` (price at the final volume bucket) for the stronger baseline. This aligns with how the literature uses VPIN: as a summary statistic across a market's activity, not a single-point snapshot.

**Evaluation:** Replaced `model.fit()` + `model.predict_proba()` on the same data with `sklearn.model_selection.cross_val_predict` (5-fold CV). Each market's predicted probability comes from a fold where it was held out of training. Wrapped in a `Pipeline(StandardScaler, LogisticRegression)` so scaling is fit inside each fold — no information leakage from test fold statistics into the scaler.

**Dual baselines:** Now report both `baseline_brier_mean` (mean price / 100) and `baseline_brier_last` (last price / 100). Skill scores reported against both. The last-price baseline is the harder test — if the model beats the market's final price assessment, that's a genuinely strong result.

**Standardized coefficients:** Coefficients divided by `scaler.scale_` so they're comparable across features. `coef_vpin` and `coef_signed_flow` can now be directly compared to see which feature contributes more.

**Housekeeping:** Moved sklearn imports to module top. Removed dead `con` parameter. Added `MIN_MARKETS = 50` as a named constant.

### Results

| Metric | Before | After |
|--------|--------|-------|
| n_markets | 86 | 86 |
| baseline_brier | 0.2110 (single) | 0.2107 (mean) / 0.2083 (last) |
| model_brier | 0.0658 (in-sample) | 0.0049 (5-fold CV) |
| skill | 68.8% (inflated) | 97.7% vs mean / 97.6% vs last |

### Why skill *increased* after fixing the bugs

Counter-intuitive: the honest out-of-sample score is better than the old in-sample score. Two reasons:

1. **Aggregation change dominates.** Full time-series mean makes `mean_signed_flow` an extremely clean separator: YES markets average +0.33, NO markets average -0.36. The old midpoint sample was noisier — a single bucket's signed flow is much less stable than the market-lifetime average.

2. **Cross-validation doesn't penalize strong signal.** CV only hurts when the model overfits to noise. When the underlying signal is strong (signed flow cleanly separates outcomes), CV confirms rather than deflates. Each 17-market test fold is predicted correctly because the pattern learned from ~69 training markets generalizes perfectly.

### Caveat

The high skill score reflects that **lifetime-aggregated signed flow** is partially forward-looking — it includes flow from late in the market when participants have increasing certainty about the outcome. This is not data leakage in the traditional sense (the model never sees test-fold labels), but it means the features encode information that wouldn't be available at a fixed prediction time (e.g., midpoint). This is an association test ("does aggregate VPIN relate to resolution?"), not a real-time prediction test.
