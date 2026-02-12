"""VPIN signal validation for Kalshi prediction markets.

Tests three claims:
1. VPIN predicts future price volatility (|price change|)
2. Signed order flow predicts future price direction
3. VPIN in first half of market life predicts YES/NO resolution

This is the first application of VPIN to prediction market data. The key
methodological advantage over equities: exact trade classification via
taker_side eliminates the Lee-Ready noise that plagues equity VPIN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.util.vpin import MIN_TRADES, vpin_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

MIN_MARKETS = 50


class VPINSignalAnalysis(Analysis):
    """Validate whether VPIN predicts price moves in Kalshi markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        bucket_size: int = 200,
        lookback: int = 10,
    ):
        super().__init__(
            name="vpin_signal",
            description="VPIN signal validation: volatility, direction, and resolution prediction",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.bucket_size = bucket_size
        self.lookback = lookback

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing VPIN across all qualified markets"):
            vpin_df = self._compute_vpin(con)

        results: dict[str, Any] = {}

        with self.progress("Testing volatility prediction"):
            vol_results, vol_quintiles = self._test_volatility_prediction(vpin_df)
            results["volatility"] = vol_results

        with self.progress("Testing directional prediction"):
            dir_results = self._test_directional_prediction(vpin_df)
            results["direction"] = dir_results

        with self.progress("Testing resolution prediction"):
            res_results = self._test_resolution_prediction(vpin_df)
            results["resolution"] = res_results

        fig = self._create_figure(vpin_df, vol_quintiles, results)
        chart = self._create_chart(vol_quintiles)

        summary_rows = []
        for test_name, test_results in results.items():
            for metric, value in test_results.items():
                summary_rows.append({"test": test_name, "metric": metric, "value": value})

        return AnalysisOutput(figure=fig, data=pd.DataFrame(summary_rows), chart=chart)

    def _compute_vpin(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Load trades, filter to qualifying markets, compute VPIN series."""
        return con.execute(
            f"""
            WITH market_info AS (
                SELECT ticker, result, event_ticker, close_time
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized' AND result IN ('yes', 'no')
            ),
            qualified_markets AS (
                SELECT m.ticker
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN market_info m ON t.ticker = m.ticker
                GROUP BY m.ticker
                HAVING SUM(t.count) >= {MIN_TRADES}
            ),
            trades AS (
                SELECT t.ticker, t.count, t.taker_side, t.yes_price, t.created_time
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN qualified_markets q ON t.ticker = q.ticker
            ),
            {vpin_cte("trades", self.bucket_size, self.lookback)}
            SELECT
                vs.*,
                m.result,
                m.event_ticker,
                m.close_time
            FROM vpin_series vs
            INNER JOIN market_info m ON vs.ticker = m.ticker
            WHERE vs.window_size = {self.lookback}
            """
        ).df()

    def _test_volatility_prediction(self, vpin_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
        """Test 1: Does VPIN predict future |price change|?"""
        results = {}

        for k in [1, 5, 10]:
            # Future absolute price change k buckets ahead
            vpin_df[f"future_price_{k}"] = vpin_df.groupby("ticker")["avg_price"].shift(-k)
            vpin_df[f"abs_change_{k}"] = (vpin_df[f"future_price_{k}"] - vpin_df["avg_price"]).abs()

            valid = vpin_df[[f"abs_change_{k}", "vpin"]].dropna()
            if len(valid) < 100:
                continue

            slope, intercept, r, p, se = stats.linregress(valid["vpin"], valid[f"abs_change_{k}"])
            results[f"vol_beta_k{k}"] = float(slope)
            results[f"vol_pvalue_k{k}"] = float(p)
            results[f"vol_r2_k{k}"] = float(r**2)

        # Quintile analysis at k=5
        valid_k5 = vpin_df[["vpin", "abs_change_5"]].dropna()
        valid_k5["quintile"] = pd.qcut(valid_k5["vpin"], 5, labels=False, duplicates="drop") + 1
        quintiles = (
            valid_k5.groupby("quintile")
            .agg(
                mean_vpin=("vpin", "mean"),
                mean_volatility=("abs_change_5", "mean"),
                n=("vpin", "count"),
            )
            .reset_index()
        )

        return results, quintiles

    def _test_directional_prediction(self, vpin_df: pd.DataFrame) -> dict[str, float]:
        """Test 2: Does signed flow predict future price direction?"""
        results = {}

        for k in [1, 5, 10]:
            vpin_df[f"future_return_{k}"] = vpin_df.groupby("ticker")["avg_price"].shift(-k) - vpin_df["avg_price"]

            valid = vpin_df[[f"future_return_{k}", "signed_flow"]].dropna()
            if len(valid) < 100:
                continue

            slope, intercept, r, p, se = stats.linregress(valid["signed_flow"], valid[f"future_return_{k}"])
            results[f"dir_beta_k{k}"] = float(slope)
            results[f"dir_pvalue_k{k}"] = float(p)
            results[f"dir_r2_k{k}"] = float(r**2)

        return results

    def _test_resolution_prediction(self, vpin_df: pd.DataFrame) -> dict[str, float]:
        """Test 3: Does VPIN in the first half of market life predict YES/NO outcome?"""
        # Truncate to first 50% of each market's volume buckets.
        # Late-market flow reflects outcome certainty — using it would be tautological.
        bucket_counts = vpin_df.groupby("ticker")["bucket_id"].transform("count")
        bucket_rank = vpin_df.groupby("ticker")["bucket_id"].rank(method="first")
        early_df = vpin_df[bucket_rank <= (bucket_counts / 2)].copy()

        # Price at the cutoff point (last bucket in the early half)
        cutoff_idx = early_df.groupby("ticker")["bucket_id"].idxmax()
        cutoff_prices = early_df.loc[cutoff_idx, ["ticker", "avg_price"]].rename(
            columns={"avg_price": "cutoff_price"}
        )

        market_agg = (
            early_df.groupby("ticker")
            .agg(
                mean_vpin=("vpin", "mean"),
                mean_signed_flow=("signed_flow", "mean"),
                mean_price=("avg_price", "mean"),
                result=("result", "first"),
            )
            .reset_index()
            .merge(cutoff_prices, on="ticker")
        )
        market_agg["y"] = (market_agg["result"] == "yes").astype(int)

        features = market_agg[["mean_price", "mean_vpin", "mean_signed_flow"]].values
        target = market_agg["y"].values

        mask = np.isfinite(features).all(axis=1)
        features = features[mask]
        target = target[mask]

        mean_price_prob = market_agg.loc[mask, "mean_price"].values / 100.0
        cutoff_price_prob = market_agg.loc[mask, "cutoff_price"].values / 100.0

        baseline_brier_mean = float(((mean_price_prob - target) ** 2).mean())
        baseline_brier_cutoff = float(((cutoff_price_prob - target) ** 2).mean())

        if len(features) < MIN_MARKETS:
            return {"n_markets": len(features), "baseline_brier_mean": baseline_brier_mean}

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ])
        model_probs = cross_val_predict(
            pipe, features, target, cv=5, method="predict_proba"
        )[:, 1]
        model_brier = float(((model_probs - target) ** 2).mean())

        # Refit on full data for coefficient extraction
        pipe.fit(features, target)
        scaler = pipe.named_steps["scaler"]
        model = pipe.named_steps["clf"]

        return {
            "n_markets": int(len(features)),
            "baseline_brier_mean": baseline_brier_mean,
            "baseline_brier_cutoff": baseline_brier_cutoff,
            "model_brier": model_brier,
            "brier_skill_vs_mean_pct": float(
                (baseline_brier_mean - model_brier) / baseline_brier_mean * 100
            ),
            "brier_skill_vs_cutoff_pct": float(
                (baseline_brier_cutoff - model_brier) / baseline_brier_cutoff * 100
            ),
            "coef_price": float(model.coef_[0][0] / scaler.scale_[0]),
            "coef_vpin": float(model.coef_[0][1] / scaler.scale_[1]),
            "coef_signed_flow": float(model.coef_[0][2] / scaler.scale_[2]),
        }

    def _create_figure(
        self,
        vpin_df: pd.DataFrame,
        quintiles: pd.DataFrame,
        results: dict[str, Any],
    ) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: VPIN vs future volatility scatter (k=5)
        ax = axes[0, 0]
        valid = vpin_df[["vpin", "abs_change_5"]].dropna()
        if len(valid) > 5000:
            sample = valid.sample(5000, random_state=42)
        else:
            sample = valid
        ax.scatter(sample["vpin"], sample["abs_change_5"], alpha=0.1, s=3, color="#3498db")
        if len(valid) > 10:
            z = np.polyfit(valid["vpin"], valid["abs_change_5"], 1)
            x_line = np.linspace(valid["vpin"].min(), valid["vpin"].max(), 100)
            ax.plot(x_line, z[0] * x_line + z[1], color="#e74c3c", linewidth=2)
        ax.set_xlabel("VPIN")
        ax.set_ylabel("|Price Change| (cents, k=5)")
        ax.set_title("VPIN vs Future Volatility")
        ax.grid(True, alpha=0.3)

        # Panel 2: Quintile bar chart
        ax = axes[0, 1]
        ax.bar(quintiles["quintile"], quintiles["mean_volatility"], color="#2ecc71", alpha=0.8)
        ax.set_xlabel("VPIN Quintile")
        ax.set_ylabel("Mean |Price Change| (cents)")
        ax.set_title("Volatility by VPIN Quintile (k=5)")
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 3: Signed flow vs future return (k=5)
        ax = axes[1, 0]
        valid_dir = vpin_df[["signed_flow", "future_return_5"]].dropna()
        if len(valid_dir) > 5000:
            sample_dir = valid_dir.sample(5000, random_state=42)
        else:
            sample_dir = valid_dir
        ax.scatter(sample_dir["signed_flow"], sample_dir["future_return_5"], alpha=0.1, s=3, color="#9b59b6")
        if len(valid_dir) > 10:
            z = np.polyfit(valid_dir["signed_flow"], valid_dir["future_return_5"], 1)
            x_line = np.linspace(valid_dir["signed_flow"].min(), valid_dir["signed_flow"].max(), 100)
            ax.plot(x_line, z[0] * x_line + z[1], color="#e74c3c", linewidth=2)
        ax.set_xlabel("Signed Flow")
        ax.set_ylabel("Future Return (cents, k=5)")
        ax.set_title("Signed Flow vs Future Direction")
        ax.grid(True, alpha=0.3)

        # Panel 4: Summary statistics table
        ax = axes[1, 1]
        ax.axis("off")
        table_data = []

        vol = results.get("volatility", {})
        for k in [1, 5, 10]:
            beta = vol.get(f"vol_beta_k{k}")
            pval = vol.get(f"vol_pvalue_k{k}")
            r2 = vol.get(f"vol_r2_k{k}")
            if beta is not None:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                table_data.append([f"VPIN → |ΔP| (k={k})", f"{beta:.3f}{sig}", f"{r2:.4f}"])

        dir_r = results.get("direction", {})
        for k in [1, 5, 10]:
            beta = dir_r.get(f"dir_beta_k{k}")
            pval = dir_r.get(f"dir_pvalue_k{k}")
            r2 = dir_r.get(f"dir_r2_k{k}")
            if beta is not None:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                table_data.append([f"Flow → ΔP (k={k})", f"{beta:.3f}{sig}", f"{r2:.4f}"])

        res = results.get("resolution", {})
        if "brier_skill_vs_mean_pct" in res:
            table_data.append(["Brier (mean price)", f"{res['baseline_brier_mean']:.4f}", ""])
            table_data.append(["Brier (cutoff price)", f"{res['baseline_brier_cutoff']:.4f}", ""])
            table_data.append(["Brier (model CV)", f"{res['model_brier']:.4f}", ""])
            table_data.append(["Skill vs mean", f"{res['brier_skill_vs_mean_pct']:.1f}%", ""])
            table_data.append(["Skill vs cutoff", f"{res['brier_skill_vs_cutoff_pct']:.1f}%", ""])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=["Test", "β / Score", "R²"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        ax.set_title("Regression Results", fontsize=12, fontweight="bold", pad=20)

        fig.suptitle(
            f"VPIN Signal Validation (bucket={self.bucket_size}, lookback={self.lookback})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_chart(self, quintiles: pd.DataFrame) -> ChartConfig:
        chart_data = [
            {
                "quintile": int(row["quintile"]),
                "Mean Volatility": round(row["mean_volatility"], 3),
                "Mean VPIN": round(row["mean_vpin"], 4),
            }
            for _, row in quintiles.iterrows()
        ]
        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="quintile",
            yKeys=["Mean Volatility"],
            title="Future Volatility by VPIN Quintile",
            yUnit=UnitType.CENTS,
            xLabel="VPIN Quintile (1=Low, 5=High)",
            yLabel="Mean |Price Change| (cents)",
        )
