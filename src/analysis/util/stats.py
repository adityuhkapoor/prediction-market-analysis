"""Reusable statistical test functions for VPIN insider trading analysis.

All functions return structured results (named tuples or dataclasses) with
p-values, effect sizes, and test metadata. Designed to be collected and
passed to ``apply_bh_fdr()`` for multiple testing correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sp_stats
from scipy.stats import false_discovery_control

# ── Result types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KSResult:
    statistic: float
    pvalue: float
    significant: bool


@dataclass(frozen=True)
class MannWhitneyResult:
    statistic: float
    pvalue: float
    significant: bool


@dataclass(frozen=True)
class PermutationResult:
    observed_stat: float
    pvalue: float
    null_distribution: np.ndarray


@dataclass(frozen=True)
class BinomialResult:
    pvalue: float
    cohen_h: float
    hit_rate: float
    hits: int
    trials: int


@dataclass(frozen=True)
class BootstrapPeakLagResult:
    peak_lag: int
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class FDRResult:
    adjusted_pvalues: np.ndarray
    significant_mask: np.ndarray


# ── Test implementations ─────────────────────────────────────────────────


def ks_test(
    sample_a: ArrayLike,
    sample_b: ArrayLike,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> KSResult:
    """Two-sample Kolmogorov-Smirnov test.

    Tests whether sample_a is stochastically greater than sample_b
    (default alternative='greater').

    Args:
        sample_a: Test sample (e.g., insider-active VPIN).
        sample_b: Reference sample (e.g., non-insider VPIN).
        alpha: Significance level.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        KSResult with D statistic, p-value, and significance flag.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    result = sp_stats.ks_2samp(a, b, alternative=alternative)
    return KSResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        significant=result.pvalue < alpha,
    )


def mann_whitney_test(
    sample_a: ArrayLike,
    sample_b: ArrayLike,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> MannWhitneyResult:
    """Mann-Whitney U test for comparing two independent samples.

    Args:
        sample_a: Test sample.
        sample_b: Reference sample.
        alpha: Significance level.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        MannWhitneyResult with U statistic, p-value, and significance flag.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    result = sp_stats.mannwhitneyu(a, b, alternative=alternative)
    return MannWhitneyResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        significant=result.pvalue < alpha,
    )


def permutation_test(
    sample_a: ArrayLike,
    sample_b: ArrayLike,
    stat_fn=None,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> PermutationResult:
    """Permutation test comparing two samples.

    Shuffles the combined pool and recomputes the test statistic to build
    an exact null distribution.

    Args:
        sample_a: Test sample.
        sample_b: Reference sample.
        stat_fn: Function(a, b) → scalar. Defaults to difference of means.
        n_permutations: Number of permutation iterations.
        seed: RNG seed for reproducibility.

    Returns:
        PermutationResult with observed statistic, p-value, and null distribution.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    def _diff_of_means(x, y):
        return np.mean(x) - np.mean(y)

    if stat_fn is None:
        stat_fn = _diff_of_means

    observed = stat_fn(a, b)
    combined = np.concatenate([a, b])
    n_a = len(a)

    rng = np.random.default_rng(seed)
    null_dist = np.empty(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(combined)
        null_dist[i] = stat_fn(combined[:n_a], combined[n_a:])

    # One-sided p-value: proportion of null >= observed
    pvalue = float(np.mean(null_dist >= observed))

    return PermutationResult(
        observed_stat=float(observed),
        pvalue=pvalue,
        null_distribution=null_dist,
    )


def binomial_test(
    hits: int,
    trials: int,
    p0: float = 0.5,
) -> BinomialResult:
    """Binomial test for hit rate against a null probability.

    Args:
        hits: Number of successes.
        trials: Total number of trials.
        p0: Null hypothesis probability.

    Returns:
        BinomialResult with p-value, Cohen's h effect size, and hit rate.
    """
    result = sp_stats.binomtest(hits, trials, p0, alternative="greater")
    hit_rate = hits / trials if trials > 0 else 0.0

    # Cohen's h: 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p0))
    cohen_h = 2 * asin(sqrt(hit_rate)) - 2 * asin(sqrt(p0))

    return BinomialResult(
        pvalue=float(result.pvalue),
        cohen_h=float(cohen_h),
        hit_rate=float(hit_rate),
        hits=hits,
        trials=trials,
    )


def bootstrap_peak_lag(
    x: ArrayLike,
    y: ArrayLike,
    max_lag: int = 20,
    n_bootstrap: int = 1_000,
    seed: int = 42,
) -> BootstrapPeakLagResult:
    """Find peak lag in cross-correlation and bootstrap a confidence interval.

    Computes the cross-correlation between x and y at lags -max_lag to +max_lag,
    finds the lag with maximum absolute correlation, then bootstraps to get
    a CI for the peak lag location.

    Args:
        x: First time series (e.g., VPIN).
        y: Second time series (e.g., price volatility).
        max_lag: Maximum lag to consider in both directions.
        n_bootstrap: Number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        BootstrapPeakLagResult with peak lag and 95% CI bounds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    def _peak_lag(x_series, y_series):
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xi = x_series[:n - lag] if lag > 0 else x_series
                yi = y_series[lag:] if lag > 0 else y_series
            else:
                xi = x_series[-lag:]
                yi = y_series[:n + lag]
            if len(xi) < 3:
                correlations.append(0.0)
                continue
            r, _ = sp_stats.pearsonr(xi, yi)
            correlations.append(abs(r) if not np.isnan(r) else 0.0)
        return range(-max_lag, max_lag + 1).__getitem__(np.argmax(correlations)) if correlations else 0

    observed_peak = _peak_lag(x, y)

    rng = np.random.default_rng(seed)
    boot_peaks = np.empty(n_bootstrap, dtype=int)
    indices = np.arange(n)

    for i in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample_idx.sort()
        boot_peaks[i] = _peak_lag(x[sample_idx], y[sample_idx])

    ci_lower = float(np.percentile(boot_peaks, 2.5))
    ci_upper = float(np.percentile(boot_peaks, 97.5))

    return BootstrapPeakLagResult(
        peak_lag=int(observed_peak),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def apply_bh_fdr(
    pvalues: ArrayLike,
    q: float = 0.10,
) -> FDRResult:
    """Apply Benjamini-Hochberg FDR correction.

    Uses scipy's ``false_discovery_control`` which implements the BH procedure.

    Args:
        pvalues: Array of raw p-values.
        q: Target false discovery rate.

    Returns:
        FDRResult with adjusted p-values and boolean significance mask.
    """
    pvals = np.asarray(pvalues, dtype=float)

    if len(pvals) == 0:
        return FDRResult(
            adjusted_pvalues=np.array([]),
            significant_mask=np.array([], dtype=bool),
        )

    # scipy's false_discovery_control returns adjusted p-values
    adjusted = false_discovery_control(pvals, method="bh")

    return FDRResult(
        adjusted_pvalues=np.asarray(adjusted),
        significant_mask=np.asarray(adjusted) <= q,
    )
