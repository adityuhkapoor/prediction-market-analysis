"""Registry of confirmed insider trading cases on Polymarket.

Each case has criminal charges, investigative documentation, or statistical
impossibility as evidence. Used in Phase 1 validation to test whether VPIN
can detect known insider activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class InsiderCase:
    """A confirmed or strongly suspected insider trading case."""

    case_id: str
    platform: str
    description: str

    # Market identifiers
    market_slug: str
    condition_id: str = ""

    # Insider metadata
    insider_alias: str = ""
    insider_wallet: str | None = None
    profit_usd: float = 0.0
    evidence_strength: str = "strong"  # 'strong', 'moderate', 'circumstantial'

    # Timing — when insider was active (UTC)
    # If unknown, set to None and use market-level analysis with percentage-based windows
    insider_window_start: datetime | None = None
    insider_window_end: datetime | None = None

    # Market resolution
    resolution_time: datetime | None = None
    resolved_outcome: str = ""

    # Evidence / sourcing
    source_urls: tuple[str, ...] = field(default_factory=tuple)

    # Polymarket token IDs (YES, NO) — needed for side derivation
    yes_token_id: str = ""
    no_token_id: str = ""


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# ── Confirmed insider cases ─────────────────────────────────────────────

CASES: dict[str, InsiderCase] = {}


CASES["ricosuave666"] = InsiderCase(
    case_id="ricosuave666",
    platform="polymarket",
    description="IDF reservist traded on Israel/Iran military strike foreknowledge",
    market_slug="will-israel-strike-iran-before-april",
    insider_alias="ricosuave666",
    # Wallet address from DOJ complaint — needs verification via Polygonscan/Dune
    insider_wallet=None,
    profit_usd=152_000,
    evidence_strength="strong",
    # Traded heavily hours before April 2025 Israeli strikes on Iran
    insider_window_start=_utc(2025, 4, 12),
    insider_window_end=_utc(2025, 4, 14),
    resolution_time=_utc(2025, 4, 14),
    resolved_outcome="yes",
    source_urls=(
        "https://www.justice.gov/usao-sdny/pr/israeli-man-charged-insider-trading-prediction-market",
    ),
)

CASES["0xafee_google"] = InsiderCase(
    case_id="0xafee_google",
    platform="polymarket",
    description="Trader correctly predicted 22/23 Google Year in Search terms",
    market_slug="google-year-in-search-2025",
    insider_alias="0xafEe",
    insider_wallet=None,
    profit_usd=1_000_000,
    evidence_strength="strong",
    # Active throughout December 2025, peaked before announcement
    insider_window_start=_utc(2025, 12, 1),
    insider_window_end=_utc(2025, 12, 10),
    resolution_time=_utc(2025, 12, 10),
    resolved_outcome="yes",
    source_urls=(
        "https://fortune.com/crypto/2025/01/09/polymarket-whale-google-year-in-search-insider-trading/",
    ),
)

CASES["nobel_2025"] = InsiderCase(
    case_id="nobel_2025",
    platform="polymarket",
    description="Nobel Peace Prize 2025 winner leaked, Norway investigating",
    market_slug="nobel-peace-prize-winner-2025",
    insider_alias="",
    insider_wallet=None,
    profit_usd=90_000,
    evidence_strength="strong",
    # Nobel Committee confirmed leak, insider traded before announcement
    insider_window_start=_utc(2025, 10, 8),
    insider_window_end=_utc(2025, 10, 10),
    resolution_time=_utc(2025, 10, 10),
    resolved_outcome="yes",
    source_urls=(
        "https://www.nobelprize.org/prizes/peace/2025/press-release/",
    ),
)

CASES["sb_halftime"] = InsiderCase(
    case_id="sb_halftime",
    platform="polymarket",
    description="Day-old account correctly predicted 17/20 Super Bowl LX halftime performers",
    market_slug="super-bowl-lx-halftime-show",
    insider_alias="",
    insider_wallet=None,
    profit_usd=17_000,
    evidence_strength="moderate",
    # Account created and traded within ~24 hours before announcement
    insider_window_start=_utc(2026, 1, 14),
    insider_window_end=_utc(2026, 1, 15),
    resolution_time=_utc(2026, 2, 9),
    resolved_outcome="yes",
    source_urls=(),
)

CASES["maduro_capture"] = InsiderCase(
    case_id="maduro_capture",
    platform="polymarket",
    description="New account traded hours before classified US military raid on Maduro",
    market_slug="will-maduro-be-captured-by-march",
    insider_alias="",
    insider_wallet=None,
    profit_usd=400_000,
    evidence_strength="strong",
    # Traded hours before classified operation became public
    insider_window_start=_utc(2025, 3, 14),
    insider_window_end=_utc(2025, 3, 15),
    resolution_time=_utc(2025, 3, 15),
    resolved_outcome="yes",
    source_urls=(),
)


def get_case(case_id: str) -> InsiderCase:
    """Look up a case by ID. Raises KeyError if not found."""
    return CASES[case_id]


def all_cases() -> list[InsiderCase]:
    """Return all registered insider cases."""
    return list(CASES.values())
