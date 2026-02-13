"""Tests for category mapping utilities."""

from src.analysis.util.categories import GROUP_COLORS, get_group, get_hierarchy


def test_known_sport():
    assert get_group("NFLGAME-xyz") == "Sports"


def test_known_politics():
    assert get_group("PRES-xyz") == "Politics"


def test_known_crypto():
    assert get_group("BTC-xyz") == "Crypto"


def test_unknown():
    assert get_group("XYZUNKNOWN") == "Other"


def test_none():
    assert get_group(None) == "Other"


def test_empty():
    assert get_group("") == "Other"


def test_hierarchy_tuple():
    result = get_hierarchy("NFLGAME-xyz")
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    assert all(isinstance(s, str) for s in result), "All elements should be strings"
    assert result[0] == "Sports"


def test_group_colors_coverage():
    """Every group returned by known tickers should exist in GROUP_COLORS."""
    known_tickers = [
        "NFLGAME-123", "PRES-456", "BTC-789", "INX-101",
        "HIGHNY-202", "SPOTIFY-303", "LLM-404", "POPE-505",
    ]
    for ticker in known_tickers:
        group = get_group(ticker)
        assert group in GROUP_COLORS, f"Group '{group}' from ticker '{ticker}' not in GROUP_COLORS"
