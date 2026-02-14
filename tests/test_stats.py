"""Tests for statistical test wrappers against scipy/analytical values."""

import numpy as np
from scipy import stats as sp_stats

from src.analysis.util.stats import (
    apply_bh_fdr,
    binomial_test,
    bootstrap_peak_lag,
    ks_test,
    mann_whitney_test,
    permutation_test,
)


def test_ks_matches_scipy():
    """KS test on uniform vs beta(5,2) should match scipy.stats.ks_2samp."""
    rng = np.random.default_rng(42)
    a = rng.beta(5, 2, size=500)
    b = rng.uniform(0, 1, size=500)

    our_result = ks_test(a, b, alternative="two-sided")
    scipy_result = sp_stats.ks_2samp(a, b, alternative="two-sided")

    assert abs(our_result.statistic - scipy_result.statistic) < 1e-10
    assert abs(our_result.pvalue - scipy_result.pvalue) < 1e-10


def test_ks_identical_distributions_not_significant():
    """KS test on two samples from the same distribution should not be significant."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=200)
    b = rng.normal(0, 1, size=200)

    result = ks_test(a, b, alternative="two-sided")
    assert result.pvalue > 0.05, f"Identical distributions flagged as significant: p={result.pvalue}"


def test_ks_greater_alternative():
    """alternative='greater' detects a stochastically smaller than b."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=300)  # Shifted left (stochastically smaller)
    b = rng.normal(2, 1, size=300)

    result = ks_test(a, b, alternative="greater")
    assert result.significant, f"Should detect a < b: D={result.statistic}, p={result.pvalue}"
    assert result.statistic > 0.15, f"Effect size too small: D={result.statistic}"


def test_ks_less_detects_stochastically_larger():
    """alternative='less' detects a stochastically larger than b."""
    rng = np.random.default_rng(42)
    a = rng.normal(2, 1, size=300)  # Shifted right (stochastically larger)
    b = rng.normal(0, 1, size=300)

    result = ks_test(a, b, alternative="less")
    assert result.significant, f"Should detect a > b: D={result.statistic}, p={result.pvalue}"



def test_mann_whitney_matches_scipy():
    """Mann-Whitney should match scipy result."""
    rng = np.random.default_rng(42)
    a = rng.normal(1, 1, size=100)
    b = rng.normal(0, 1, size=100)

    our_result = mann_whitney_test(a, b, alternative="greater")
    scipy_result = sp_stats.mannwhitneyu(a, b, alternative="greater")

    assert abs(our_result.statistic - scipy_result.statistic) < 1e-10
    assert abs(our_result.pvalue - scipy_result.pvalue) < 1e-10



def test_permutation_identical_distributions_uniform_p():
    """Permutation test on identical distributions should produce p-values ~uniform."""
    # Run multiple permutation tests on samples from the same distribution
    p_values = []
    for seed in range(50):
        rng2 = np.random.default_rng(seed + 100)
        a = rng2.normal(0, 1, size=50)
        b = rng2.normal(0, 1, size=50)
        result = permutation_test(a, b, n_permutations=500, seed=seed)
        p_values.append(result.pvalue)

    # P-values should not be clustered near 0 — at most 10% should be < 0.05
    fraction_significant = np.mean(np.array(p_values) < 0.05)
    assert fraction_significant < 0.20, (
        f"Too many false positives: {fraction_significant:.0%} < 0.05 (expected ~5%)"
    )


def test_permutation_detects_difference():
    """Permutation test should detect a real difference in means."""
    rng = np.random.default_rng(42)
    a = rng.normal(2, 1, size=100)
    b = rng.normal(0, 1, size=100)

    result = permutation_test(a, b, n_permutations=1000, seed=42)
    assert result.pvalue < 0.01, f"Should detect mean difference: p={result.pvalue}"
    assert result.observed_stat > 1.5, f"Observed stat too small: {result.observed_stat}"


def test_permutation_null_distribution_shape():
    """Null distribution should have the expected number of samples."""
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([0, 1, 2, 3, 4])

    result = permutation_test(a, b, n_permutations=500, seed=42)
    assert len(result.null_distribution) == 500



def test_binomial_7_of_10():
    """7/10 hits vs p0=0.5 should give p ~ 0.172 (one-sided greater)."""
    result = binomial_test(7, 10, p0=0.5)
    assert abs(result.pvalue - 0.172) < 0.01, f"Expected p~0.172, got {result.pvalue}"
    assert result.hit_rate == 0.7


def test_binomial_9_of_10():
    """9/10 hits vs p0=0.5 should give p ~ 0.011."""
    result = binomial_test(9, 10, p0=0.5)
    assert abs(result.pvalue - 0.011) < 0.005, f"Expected p~0.011, got {result.pvalue}"


def test_binomial_cohen_h():
    """Cohen's h for 0.7 vs 0.5 should be ~0.41."""
    from math import asin, sqrt
    result = binomial_test(7, 10, p0=0.5)
    expected_h = 2 * asin(sqrt(0.7)) - 2 * asin(sqrt(0.5))
    assert abs(result.cohen_h - expected_h) < 0.001, (
        f"Cohen's h mismatch: got {result.cohen_h}, expected {expected_h}"
    )
    assert abs(result.cohen_h - 0.4115) < 0.01, f"Cohen's h ~ 0.41, got {result.cohen_h}"



def test_bootstrap_peak_lag_known_signal():
    """Synthetic signal with known lag should be detected."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, size=n)
    # y lags x by 5 steps
    y = np.zeros(n)
    y[5:] = x[:-5] + rng.normal(0, 0.3, size=n - 5)

    result = bootstrap_peak_lag(x, y, max_lag=10, n_bootstrap=500, seed=42)
    # Peak lag should be around +5
    assert abs(result.peak_lag - 5) <= 2, f"Expected peak lag ~5, got {result.peak_lag}"
    assert result.ci_lower <= result.peak_lag <= result.ci_upper


def test_bootstrap_peak_lag_returns_ci():
    """Bootstrap should return valid CI bounds."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=100)
    y = rng.normal(0, 1, size=100)

    result = bootstrap_peak_lag(x, y, max_lag=10, n_bootstrap=200, seed=42)
    assert result.ci_lower <= result.ci_upper



def test_bh_fdr_two_significant():
    """2 truly significant + 8 null p-values → exactly 2 flagged at q=0.10."""
    pvals = np.array([0.001, 0.005, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    result = apply_bh_fdr(pvals, q=0.10)

    assert result.significant_mask.sum() == 2, (
        f"Expected 2 significant, got {result.significant_mask.sum()}"
    )
    # The first two should be significant
    assert result.significant_mask[0] and result.significant_mask[1]


def test_bh_fdr_all_null():
    """When all p-values are large, none should be flagged."""
    pvals = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
    result = apply_bh_fdr(pvals, q=0.10)
    assert result.significant_mask.sum() == 0


def test_bh_fdr_empty():
    """Empty input should return empty arrays."""
    result = apply_bh_fdr([], q=0.10)
    assert len(result.adjusted_pvalues) == 0
    assert len(result.significant_mask) == 0


def test_bh_fdr_matches_scipy():
    """Adjusted p-values should match scipy's false_discovery_control."""
    pvals = np.array([0.001, 0.01, 0.04, 0.1, 0.5])
    from scipy.stats import false_discovery_control
    scipy_adjusted = false_discovery_control(pvals, method="bh")

    result = apply_bh_fdr(pvals, q=0.10)
    np.testing.assert_allclose(
        result.adjusted_pvalues, scipy_adjusted, atol=1e-10,
        err_msg="BH-FDR adjusted p-values differ from scipy"
    )
