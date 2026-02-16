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
    rng = np.random.default_rng(42)
    a = rng.beta(5, 2, size=500)
    b = rng.uniform(0, 1, size=500)

    our = ks_test(a, b, alternative="two-sided")
    ref = sp_stats.ks_2samp(a, b, alternative="two-sided")

    assert abs(our.statistic - ref.statistic) < 1e-10
    assert abs(our.pvalue - ref.pvalue) < 1e-10


def test_ks_identical_not_significant():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=200)
    b = rng.normal(0, 1, size=200)
    assert ks_test(a, b, alternative="two-sided").pvalue > 0.05


def test_ks_greater_alternative():
    """alternative='greater' detects a stochastically smaller than b."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=300)
    b = rng.normal(2, 1, size=300)

    result = ks_test(a, b, alternative="greater")
    assert result.significant
    assert result.statistic > 0.15


def test_ks_less_detects_stochastically_larger():
    """alternative='less' detects a stochastically larger than b."""
    rng = np.random.default_rng(42)
    a = rng.normal(2, 1, size=300)
    b = rng.normal(0, 1, size=300)
    assert ks_test(a, b, alternative="less").significant


def test_mann_whitney_matches_scipy():
    rng = np.random.default_rng(42)
    a = rng.normal(1, 1, size=100)
    b = rng.normal(0, 1, size=100)

    our = mann_whitney_test(a, b, alternative="greater")
    ref = sp_stats.mannwhitneyu(a, b, alternative="greater")

    assert abs(our.statistic - ref.statistic) < 1e-10
    assert abs(our.pvalue - ref.pvalue) < 1e-10


def test_permutation_null_pvalues_not_clustered():
    p_values = []
    for seed in range(50):
        rng = np.random.default_rng(seed + 100)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(0, 1, size=50)
        result = permutation_test(a, b, n_permutations=500, seed=seed)
        p_values.append(result.pvalue)

    frac_sig = np.mean(np.array(p_values) < 0.05)
    assert frac_sig < 0.20, f"Too many false positives: {frac_sig:.0%}"


def test_permutation_detects_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(2, 1, size=100)
    b = rng.normal(0, 1, size=100)

    result = permutation_test(a, b, n_permutations=1000, seed=42)
    assert result.pvalue < 0.01
    assert result.observed_stat > 1.5


def test_permutation_null_distribution_length():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([0, 1, 2, 3, 4])
    assert len(permutation_test(a, b, n_permutations=500, seed=42).null_distribution) == 500


def test_binomial_7_of_10():
    result = binomial_test(7, 10, p0=0.5)
    assert abs(result.pvalue - 0.172) < 0.01
    assert result.hit_rate == 0.7


def test_binomial_9_of_10():
    result = binomial_test(9, 10, p0=0.5)
    assert abs(result.pvalue - 0.011) < 0.005


def test_binomial_cohen_h():
    """Cohen's h for 0.7 vs 0.5 should be ~0.41."""
    from math import asin, sqrt

    result = binomial_test(7, 10, p0=0.5)
    expected = 2 * asin(sqrt(0.7)) - 2 * asin(sqrt(0.5))
    assert abs(result.cohen_h - expected) < 0.001
    assert abs(result.cohen_h - 0.4115) < 0.01


def test_bootstrap_peak_lag_known_signal():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, size=n)
    y = np.zeros(n)
    y[5:] = x[:-5] + rng.normal(0, 0.3, size=n - 5)

    result = bootstrap_peak_lag(x, y, max_lag=10, n_bootstrap=500, seed=42)
    assert abs(result.peak_lag - 5) <= 2, f"Expected peak lag ~5, got {result.peak_lag}"
    assert result.ci_lower <= result.peak_lag <= result.ci_upper


def test_bootstrap_peak_lag_ci_bounds():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=100)
    y = rng.normal(0, 1, size=100)
    result = bootstrap_peak_lag(x, y, max_lag=10, n_bootstrap=200, seed=42)
    assert result.ci_lower <= result.ci_upper


def test_bh_fdr_two_significant():
    pvals = np.array([0.001, 0.005, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    result = apply_bh_fdr(pvals, q=0.10)
    assert result.significant_mask.sum() == 2
    assert result.significant_mask[0] and result.significant_mask[1]


def test_bh_fdr_all_null():
    pvals = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
    assert apply_bh_fdr(pvals, q=0.10).significant_mask.sum() == 0


def test_bh_fdr_empty():
    result = apply_bh_fdr([], q=0.10)
    assert len(result.adjusted_pvalues) == 0
    assert len(result.significant_mask) == 0


def test_bh_fdr_matches_scipy():
    pvals = np.array([0.001, 0.01, 0.04, 0.1, 0.5])
    from scipy.stats import false_discovery_control

    expected = false_discovery_control(pvals, method="bh")
    result = apply_bh_fdr(pvals, q=0.10)
    np.testing.assert_allclose(result.adjusted_pvalues, expected, atol=1e-10)
