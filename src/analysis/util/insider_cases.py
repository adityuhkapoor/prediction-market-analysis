"""Registry of confirmed insider trading cases on Polymarket."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class InsiderCase:
    case_id: str
    platform: str
    description: str
    market_slug: str
    condition_id: str = ""
    insider_alias: str = ""
    insider_wallet: str | None = None
    profit_usd: float = 0.0
    evidence_strength: str = "strong"
    insider_window_start: datetime | None = None
    insider_window_end: datetime | None = None
    resolution_time: datetime | None = None
    resolved_outcome: str = ""
    source_urls: tuple[str, ...] = field(default_factory=tuple)
    yes_token_id: str = ""
    no_token_id: str = ""


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


CASES: dict[str, InsiderCase] = {}


CASES["ricosuave666"] = InsiderCase(
    case_id="ricosuave666",
    platform="polymarket",
    description="IDF reservist traded on Israel/Iran strike foreknowledge (Operation Rising Lion)",
    market_slug="israel-military-action-against-iran-by-friday-477",
    condition_id="0x7f39808829da93cfd189807f13f6d86a0e604835e6f9482d8094fac46b3abaac",
    insider_alias="ricosuave666",
    insider_wallet="0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
    profit_usd=152_300,
    evidence_strength="strong",
    # Traded ~48h before Israel launched Operation Rising Lion (June 13, 2025)
    insider_window_start=_utc(2025, 6, 11),
    insider_window_end=_utc(2025, 6, 13),
    resolution_time=_utc(2025, 6, 13),
    resolved_outcome="yes",
    yes_token_id="19456040549238228129383854351183747737493394931814797928509171842469189313430",
    no_token_id="16808727832337773803213654645727064011490757284639067407550873885798909855144",
    source_urls=(
        "https://www.timesofisrael.com/two-indicted-for-using-classified-info-to-place-online-bets-on-military-operations/",
        "https://decrypt.co/357892/israelis-arrested-alleged-insider-polymarket-trades-idf-military-secrets",
    ),
)

CASES["0xafee_google"] = InsiderCase(
    case_id="0xafee_google",
    platform="polymarket",
    description="Trader correctly predicted 22/23 Google Year in Search terms (d4vd #1 searched)",
    market_slug="will-d4vd-rank-in-googles-top-5-most-searched-people-of-2025",
    condition_id="0xeaf59fcbf65e45abac0383dad483239d849e6d48d9eb2a6b3bf5cc1c7e9cf2ad",
    insider_alias="0xafEe",
    insider_wallet="0xee50a31c3f5a7c77824b12a941a54388a2827ed6",
    profit_usd=1_000_000,
    evidence_strength="strong",
    # Deposited $3M and traded across 20+ markets before Google published results early
    insider_window_start=_utc(2025, 11, 28),
    insider_window_end=_utc(2025, 12, 4),
    resolution_time=_utc(2025, 12, 4),
    resolved_outcome="yes",
    yes_token_id="99812311604636667753353213036457655745980182349717710552725887133141885629693",
    no_token_id="112700059580288003746967245246151029963591601924216908570415116368812396052049",
    source_urls=(
        "https://finance.yahoo.com/news/polymarket-trader-makes-1-million-090001027.html",
        "https://thedefiant.io/news/defi/polymarket-users-suspect-insider-trading-after-google-trend-markets-crown-surprise-winner",
    ),
)

CASES["nobel_2025"] = InsiderCase(
    case_id="nobel_2025",
    platform="polymarket",
    description="Nobel Peace Prize 2025 leak — Machado winner known before announcement",
    market_slug="will-mara-corina-machado-win-the-nobel-peace-prize-in-2025",
    condition_id="0x14a3dfeba8b22a32feb0f10763db68bc4d2abeb5bff90e9ae20de53793b35a1d",
    insider_alias="dirtycup",
    insider_wallet="0x234cc49e43dff8b3207bbd3a8a2579f339cb9867",
    profit_usd=90_000,
    evidence_strength="strong",
    # Suspicious trading Oct 9-10 UTC, announcement at 09:00 UTC Oct 10
    insider_window_start=_utc(2025, 10, 9, 17),
    insider_window_end=_utc(2025, 10, 10, 9),
    resolution_time=_utc(2025, 10, 10, 9),
    resolved_outcome="yes",
    yes_token_id="75406677405586796507165996816755205915219135539748390183994296030461533919844",
    no_token_id="97086808654593962926257938662068923166271957549978936996637041103040300675718",
    source_urls=(
        "https://finance.yahoo.com/news/norway-probes-potential-insider-trading-194041145.html",
        "https://blockworks.co/news/polymarket-nobel-probe",
    ),
)

CASES["sb_halftime"] = InsiderCase(
    case_id="sb_halftime",
    platform="polymarket",
    description="Day-old account correctly predicted 17/20 Super Bowl LX halftime performers",
    market_slug="will-lady-gaga-perform-during-the-super-bowl-lx-halftime-show",
    condition_id="0x70705ca527ecbcbdb5ebc5f648a5b0f3e765a0839d5889f6fae7e80c7e036625",
    insider_alias="",
    insider_wallet="0x40d9ac81a425f14d2c490c41ac8969c0cbcfd472",
    profit_usd=17_000,
    evidence_strength="moderate",
    # Account created ~Feb 7, traded Feb 7-9 before Super Bowl LX
    insider_window_start=_utc(2026, 2, 7),
    insider_window_end=_utc(2026, 2, 9),
    resolution_time=_utc(2026, 2, 9),
    resolved_outcome="yes",
    yes_token_id="84920169572240366196592437314195546665530796969027433493641169692446880756137",
    no_token_id="11835028779183763019515322571299156542620302261312155497321162778894726543380",
    source_urls=(
        "https://finance.yahoo.com/news/polymarket-user-wins-nearly-super-204100506.html",
        "https://www.benzinga.com/markets/prediction-markets/26/02/50471679/lady-gaga-ricky-martin-appearances-nailed-by-suspicious-insider-polymarket-trader-rais",
    ),
)

CASES["maduro_capture"] = InsiderCase(
    case_id="maduro_capture",
    platform="polymarket",
    description="New account traded hours before classified US raid on Maduro (Operation Absolute Resolve)",
    market_slug="maduro-out-by-january-31-2026-318",
    condition_id="0x580adc1327de9bf7c179ef5aaffa3377bb5cb252b7d6390b027172d43fd6f993",
    insider_alias="Burdensome-Mix",
    insider_wallet="0x31a56e9E690c621eD21De08Cb559e9524Cdb8eD9",
    profit_usd=436_000,
    evidence_strength="strong",
    # Final buy at ~02:00 UTC Jan 3, ~4h before US aircraft entered Venezuelan airspace
    insider_window_start=_utc(2026, 1, 2, 21),
    insider_window_end=_utc(2026, 1, 3, 12),
    resolution_time=_utc(2026, 1, 3, 12),
    resolved_outcome="yes",
    yes_token_id="24918067747661759048720135607687934209172945708698786072564741153847301974566",
    no_token_id="98590363695810976489755414330258191966783137178752226225745237702303580425394",
    source_urls=(
        "https://www.npr.org/2026/01/05/nx-s1-5667232/polymarket-maduro-bet-insider-trading",
        "https://fortune.com/2026/01/05/polymarket-user-400k-maduro-capture-bets-insider-trading-suspicion/",
    ),
)


def get_case(case_id: str) -> InsiderCase:
    """Look up a case by ID. Raises KeyError if not found."""
    return CASES[case_id]


def all_cases() -> list[InsiderCase]:
    """Return all registered insider cases."""
    return list(CASES.values())
