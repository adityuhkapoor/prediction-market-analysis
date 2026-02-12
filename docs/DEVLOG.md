# Devlog

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
