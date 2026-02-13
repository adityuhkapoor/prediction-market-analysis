"""VPIN decomposition by market category.

Computes mean VPIN and signal strength per category, testing the hypothesis
that categories with more retail/biased flow (sports, world events) exhibit
higher VPIN than categories with sophisticated participants (finance).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.util.categories import CATEGORY_SQL, GROUP_COLORS, get_group
from src.analysis.util.vpin import qualified_trades_cte, vpin_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType


class VPINCategoriesAnalysis(Analysis):
    """VPIN decomposed by market category on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        bucket_size: int = 200,
        lookback: int = 10,
    ):
        super().__init__(
            name="vpin_categories",
            description="VPIN and order flow toxicity by market category",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.bucket_size = bucket_size
        self.lookback = lookback

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing VPIN by category"):
            vpin_df = self._load_vpin_with_categories(con)

        self._require_data(vpin_df, "VPIN with categories")

        with self.progress("Computing per-category statistics"):
            group_df = self._compute_category_stats(vpin_df)

        with self.progress("Computing per-category signal strength"):
            group_df = self._compute_signal_strength(vpin_df, group_df)

        fig = self._create_figure(group_df)
        chart = self._create_chart(group_df)

        return AnalysisOutput(figure=fig, data=group_df, chart=chart)

    def _load_vpin_with_categories(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        df = con.execute(
            f"""
            WITH {qualified_trades_cte(self.trades_dir, self.markets_dir)},
            {vpin_cte("trades", self.bucket_size, self.lookback)}
            SELECT
                vs.*,
                {CATEGORY_SQL.replace("event_ticker", "m.event_ticker")} AS category
            FROM vpin_series vs
            INNER JOIN market_info m ON vs.ticker = m.ticker
            WHERE vs.window_size = {self.lookback}
            """
        ).df()

        df["group"] = df["category"].apply(get_group)
        return df

    def _compute_category_stats(self, vpin_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for group in vpin_df["group"].unique():
            gdata = vpin_df[vpin_df["group"] == group]
            rows.append(
                {
                    "group": group,
                    "mean_vpin": gdata["vpin"].mean(),
                    "median_vpin": gdata["vpin"].median(),
                    "mean_abs_signed_flow": gdata["signed_flow"].abs().mean(),
                    "n_markets": gdata["ticker"].nunique(),
                    "n_buckets": len(gdata),
                }
            )

        df = pd.DataFrame(rows)
        return df[df["n_markets"] >= 5].sort_values("n_buckets", ascending=False)

    def _compute_signal_strength(self, vpin_df: pd.DataFrame, group_df: pd.DataFrame) -> pd.DataFrame:
        """Per-category regression: VPIN → future |price change| at k=5."""
        # Compute future volatility
        vpin_df["future_price_5"] = vpin_df.groupby("ticker")["avg_price"].shift(-5)
        vpin_df["abs_change_5"] = (vpin_df["future_price_5"] - vpin_df["avg_price"]).abs()

        betas = {}
        for group in group_df["group"].values:
            gdata = vpin_df[vpin_df["group"] == group][["vpin", "abs_change_5"]].dropna()
            if len(gdata) < 50:
                betas[group] = np.nan
                continue
            slope, _, _, p, _ = stats.linregress(gdata["vpin"], gdata["abs_change_5"])
            betas[group] = slope if p < 0.10 else np.nan

        group_df["vol_beta"] = group_df["group"].map(betas)
        return group_df

    def _create_figure(self, group_df: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        top = group_df.head(8)
        colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in top["group"]]

        # Panel 1: Mean VPIN by category
        ax = axes[0]
        x = np.arange(len(top))
        ax.bar(x, top["mean_vpin"], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(top["group"], rotation=45, ha="right")
        ax.set_ylabel("Mean VPIN")
        ax.set_title("Order Flow Toxicity by Category")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate market counts
        for i, row in enumerate(top.itertuples()):
            ax.text(i, row.mean_vpin + 0.005, f"n={row.n_markets}", ha="center", fontsize=8)

        # Panel 2: Signal strength (beta) by category
        ax = axes[1]
        valid_beta = top.dropna(subset=["vol_beta"])
        if len(valid_beta) > 0:
            beta_colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in valid_beta["group"]]
            x2 = np.arange(len(valid_beta))
            ax.bar(x2, valid_beta["vol_beta"], color=beta_colors, alpha=0.8)
            ax.set_xticks(x2)
            ax.set_xticklabels(valid_beta["group"], rotation=45, ha="right")
        ax.set_ylabel("β (VPIN → |ΔPrice|)")
        ax.set_title("VPIN Signal Strength by Category")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"VPIN by Category (bucket={self.bucket_size}, lookback={self.lookback})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_chart(self, group_df: pd.DataFrame) -> ChartConfig:
        top = group_df.head(8)
        chart_data = [
            {
                "category": row["group"],
                "Mean VPIN": round(row["mean_vpin"], 4),
                "Markets": int(row["n_markets"]),
            }
            for _, row in top.iterrows()
        ]
        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="category",
            yKeys=["Mean VPIN"],
            title="Order Flow Toxicity by Category",
            xLabel="Category",
            yLabel="Mean VPIN",
            colors={g: GROUP_COLORS.get(g, "#aaaaaa") for g in top["group"]},
        )
