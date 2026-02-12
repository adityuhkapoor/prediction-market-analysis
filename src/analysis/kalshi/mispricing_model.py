"""Mispricing predictor using VPIN and market features.

Trains a model to predict whether a market resolves YES or NO, using
VPIN, signed order flow, price, category, and time-to-resolution as
features. Compares against the market price baseline and a longshot
bias correction.

Three models evaluated on a temporal out-of-sample test set:
  Model 0: Market price as probability (baseline)
  Model 1: Logistic regression on price alone (longshot correction)
  Model 2: Full model with VPIN + market features

Evaluation: Brier score, log-loss, calibration curves, Diebold-Mariano test.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.analysis.util.categories import GROUP_COLORS, get_group
from src.analysis.util.vpin import MIN_TRADES, vpin_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MispricingModelAnalysis(Analysis):
    """Predict market mispricing using VPIN and order flow features."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        bucket_size: int = 200,
        lookback: int = 10,
        train_frac: float = 0.7,
    ):
        super().__init__(
            name="mispricing_model",
            description="Mispricing predictor: market price vs VPIN-augmented model",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.bucket_size = bucket_size
        self.lookback = lookback
        self.train_frac = train_frac

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Extracting features at market midpoints"):
            features_df = self._extract_features(con)

        with self.progress("Splitting train/test by time"):
            train_df, test_df = self._temporal_split(features_df)

        with self.progress("Computing longshot adjustment from training set"):
            train_df, test_df = self._compute_longshot_adj(train_df, test_df)

        with self.progress("Training models"):
            model_results = self._train_and_evaluate(train_df, test_df)

        fig = self._create_figure(model_results, test_df)
        chart = self._create_chart(model_results)

        summary = self._build_summary(model_results, train_df, test_df)
        return AnalysisOutput(figure=fig, data=summary, chart=chart)

    def _extract_features(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Extract one feature row per market at its volume midpoint."""
        # Compute VPIN series, then pick the midpoint bucket per market
        vpin_df = con.execute(
            f"""
            WITH market_info AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized' AND result IN ('yes', 'no')
            ),
            trade_counts AS (
                SELECT t.ticker, SUM(t.count) AS total_contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN market_info m ON t.ticker = m.ticker
                GROUP BY t.ticker
                HAVING SUM(t.count) >= {MIN_TRADES}
            ),
            trades AS (
                SELECT t.ticker, t.count, t.taker_side, t.yes_price, t.created_time
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN trade_counts tc ON t.ticker = tc.ticker
            ),
            {vpin_cte("trades", self.bucket_size, self.lookback)},
            bucket_counts AS (
                SELECT ticker, MAX(bucket_id) AS max_bucket
                FROM vpin_series
                WHERE window_size = {self.lookback}
                GROUP BY ticker
            )
            SELECT
                vs.ticker,
                vs.bucket_id,
                vs.avg_price,
                vs.vpin,
                vs.signed_flow,
                vs.bucket_end,
                m.result,
                m.event_ticker,
                m.close_time,
                bc.max_bucket
            FROM vpin_series vs
            INNER JOIN market_info m ON vs.ticker = m.ticker
            INNER JOIN bucket_counts bc ON vs.ticker = bc.ticker
            INNER JOIN trade_counts tc ON vs.ticker = tc.ticker
            WHERE vs.window_size = {self.lookback}
            """
        ).df()

        if vpin_df.empty:
            return pd.DataFrame()

        # Pick the bucket closest to the midpoint for each market
        vpin_df["mid_target"] = vpin_df["max_bucket"] / 2.0
        vpin_df["mid_dist"] = (vpin_df["bucket_id"] - vpin_df["mid_target"]).abs()
        idx = vpin_df.groupby("ticker")["mid_dist"].idxmin()
        midpoints = vpin_df.loc[idx].copy()

        # Derived features
        midpoints["y"] = (midpoints["result"] == "yes").astype(int)
        midpoints["price"] = midpoints["avg_price"]
        midpoints["group"] = midpoints["event_ticker"].apply(get_group)
        # Volume observed at midpoint, approximated from bucket geometry.
        # bucket_id = FLOOR((cum_vol - 1) / bucket_size), so cum_vol at bucket k
        # is in [(k * bucket_size + 1), (k + 1) * bucket_size].
        midpoints["log_volume"] = np.log1p((midpoints["bucket_id"] + 1) * self.bucket_size)

        # Days to close
        midpoints["bucket_end"] = pd.to_datetime(midpoints["bucket_end"])
        midpoints["close_time"] = pd.to_datetime(midpoints["close_time"])
        midpoints["days_to_close"] = (
            (midpoints["close_time"] - midpoints["bucket_end"]).dt.total_seconds() / 86400
        ).clip(lower=0)

        return midpoints

    def _temporal_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by close_time: first train_frac by time → train, rest → test."""
        df = df.sort_values("close_time").reset_index(drop=True)
        split_idx = int(len(df) * self.train_frac)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _compute_longshot_adj(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute longshot bias adjustment from training data only."""
        train_df["price_bin"] = pd.cut(train_df["price"], bins=20, labels=False)
        bin_stats = (
            train_df.groupby("price_bin")
            .agg(
                empirical_win_rate=("y", "mean"),
                bin_mean_price=("price", "mean"),
                n=("y", "count"),
            )
            .reset_index()
        )
        bin_stats["longshot_adj"] = bin_stats["empirical_win_rate"] - bin_stats["bin_mean_price"] / 100

        # Map to train
        adj_map = dict(zip(bin_stats["price_bin"], bin_stats["longshot_adj"]))
        train_df["longshot_adj"] = train_df["price_bin"].map(adj_map).fillna(0)

        # Map to test using the same bin edges
        test_df["price_bin"] = pd.cut(test_df["price"], bins=20, labels=False)
        test_df["longshot_adj"] = test_df["price_bin"].map(adj_map).fillna(0)

        return train_df, test_df

    def _train_and_evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        results = {}
        test_y = test_df["y"].values

        # Model 0: Market price baseline
        p0 = test_df["price"].values / 100.0
        p0 = np.clip(p0, 0.01, 0.99)
        results["model0"] = {
            "name": "Market Price",
            "brier": float(((p0 - test_y) ** 2).mean()),
            "logloss": float(-np.mean(test_y * np.log(p0) + (1 - test_y) * np.log(1 - p0))),
            "probs": p0,
        }

        # Model 1: Logistic on price only (longshot correction)
        X_train_1 = train_df[["price"]].values
        X_test_1 = test_df[["price"]].values
        scaler_1 = StandardScaler()
        X_train_1s = scaler_1.fit_transform(X_train_1)
        X_test_1s = scaler_1.transform(X_test_1)

        m1 = LogisticRegression(max_iter=1000)
        m1.fit(X_train_1s, train_df["y"].values)
        p1 = np.clip(m1.predict_proba(X_test_1s)[:, 1], 0.01, 0.99)
        results["model1"] = {
            "name": "Longshot Correction",
            "brier": float(((p1 - test_y) ** 2).mean()),
            "logloss": float(-np.mean(test_y * np.log(p1) + (1 - test_y) * np.log(1 - p1))),
            "probs": p1,
        }

        # Model 2: Full model with VPIN features
        feature_cols = ["price", "vpin", "signed_flow", "days_to_close", "log_volume", "longshot_adj"]
        # Add category dummies
        all_groups = pd.concat([train_df["group"], test_df["group"]]).unique()
        for g in all_groups:
            col = f"cat_{g}"
            train_df[col] = (train_df["group"] == g).astype(int)
            test_df[col] = (test_df["group"] == g).astype(int)
            feature_cols.append(col)

        X_train_2 = train_df[feature_cols].values.astype(float)
        X_test_2 = test_df[feature_cols].values.astype(float)

        # Handle NaN/inf
        X_train_2 = np.nan_to_num(X_train_2, nan=0, posinf=0, neginf=0)
        X_test_2 = np.nan_to_num(X_test_2, nan=0, posinf=0, neginf=0)

        scaler_2 = StandardScaler()
        X_train_2s = scaler_2.fit_transform(X_train_2)
        X_test_2s = scaler_2.transform(X_test_2)

        m2 = LogisticRegression(max_iter=1000)
        m2.fit(X_train_2s, train_df["y"].values)
        p2 = np.clip(m2.predict_proba(X_test_2s)[:, 1], 0.01, 0.99)
        results["model2"] = {
            "name": "Full (VPIN + Features)",
            "brier": float(((p2 - test_y) ** 2).mean()),
            "logloss": float(-np.mean(test_y * np.log(p2) + (1 - test_y) * np.log(1 - p2))),
            "probs": p2,
            "coefs": dict(zip(feature_cols, m2.coef_[0] / scaler_2.scale_)),
        }

        # Diebold-Mariano test: Model 0 vs Model 2
        d = (p0 - test_y) ** 2 - (p2 - test_y) ** 2
        dm_stat = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else 0
        dm_pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        results["dm_test"] = {"statistic": float(dm_stat), "p_value": float(dm_pvalue)}

        results["test_y"] = test_y
        return results

    def _create_figure(self, results: dict, test_df: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        test_y = results["test_y"]

        # Panel 1: Brier score comparison
        ax = axes[0, 0]
        models = ["model0", "model1", "model2"]
        names = [results[m]["name"] for m in models]
        briers = [results[m]["brier"] for m in models]
        colors = ["#95a5a6", "#3498db", "#e74c3c"]
        bars = ax.bar(names, briers, color=colors, alpha=0.8)
        ax.set_ylabel("Brier Score (lower = better)")
        ax.set_title("Model Comparison")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, briers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{val:.4f}", ha="center", fontsize=10)

        # Panel 2: Calibration curves
        ax = axes[0, 1]
        for m_key, color in zip(models, colors):
            probs = results[m_key]["probs"]
            name = results[m_key]["name"]
            # Bin predictions into deciles
            bins = np.linspace(0, 1, 11)
            bin_idx = np.digitize(probs, bins) - 1
            bin_idx = np.clip(bin_idx, 0, 9)
            cal_x, cal_y = [], []
            for b in range(10):
                mask = bin_idx == b
                if mask.sum() > 0:
                    cal_x.append(probs[mask].mean())
                    cal_y.append(test_y[mask].mean())
            ax.plot(cal_x, cal_y, "o-", label=name, color=color, markersize=5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("Calibration Curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 3: Feature coefficients (Model 2)
        ax = axes[1, 0]
        coefs = results["model2"].get("coefs", {})
        # Show only non-category features for clarity
        core_features = ["price", "vpin", "signed_flow", "days_to_close", "log_volume", "longshot_adj"]
        core_coefs = {k: coefs[k] for k in core_features if k in coefs}
        if core_coefs:
            feat_names = list(core_coefs.keys())
            feat_vals = list(core_coefs.values())
            bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in feat_vals]
            ax.barh(feat_names, feat_vals, color=bar_colors, alpha=0.8)
            ax.set_xlabel("Coefficient (unscaled)")
            ax.set_title("Feature Importance (Model 2)")
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(True, alpha=0.3, axis="x")

        # Panel 4: Brier improvement by category
        ax = axes[1, 1]
        test_df = test_df.copy()
        test_df["sq_err_0"] = (results["model0"]["probs"] - test_y) ** 2
        test_df["sq_err_2"] = (results["model2"]["probs"] - test_y) ** 2
        cat_improvement = (
            test_df.groupby("group")
            .agg(
                brier_0=("sq_err_0", "mean"),
                brier_2=("sq_err_2", "mean"),
                n=("y", "count"),
            )
            .reset_index()
        )
        cat_improvement["improvement_pct"] = (
            (cat_improvement["brier_0"] - cat_improvement["brier_2"]) / cat_improvement["brier_0"] * 100
        )
        cat_improvement = cat_improvement[cat_improvement["n"] >= 20].sort_values("improvement_pct", ascending=True)

        if len(cat_improvement) > 0:
            cat_colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in cat_improvement["group"]]
            ax.barh(cat_improvement["group"], cat_improvement["improvement_pct"], color=cat_colors, alpha=0.8)
            ax.set_xlabel("Brier Score Improvement (%)")
            ax.set_title("Improvement by Category")
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(True, alpha=0.3, axis="x")

        dm = results["dm_test"]
        fig.suptitle(
            f"Mispricing Predictor (DM stat={dm['statistic']:.2f}, p={dm['p_value']:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def _create_chart(self, results: dict) -> ChartConfig:
        models = ["model0", "model1", "model2"]
        chart_data = [
            {
                "model": results[m]["name"],
                "Brier Score": round(results[m]["brier"], 4),
                "Log-Loss": round(results[m]["logloss"], 4),
            }
            for m in models
        ]
        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="model",
            yKeys=["Brier Score"],
            title="Mispricing Predictor: Model Comparison",
            yUnit=UnitType.NUMBER,
            xLabel="Model",
            yLabel="Brier Score (lower = better)",
        )

    def _build_summary(self, results: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        rows = [
            {"metric": "train_markets", "value": len(train_df)},
            {"metric": "test_markets", "value": len(test_df)},
        ]
        for m_key in ["model0", "model1", "model2"]:
            prefix = results[m_key]["name"]
            rows.append({"metric": f"{prefix}_brier", "value": results[m_key]["brier"]})
            rows.append({"metric": f"{prefix}_logloss", "value": results[m_key]["logloss"]})

        dm = results["dm_test"]
        rows.append({"metric": "dm_statistic", "value": dm["statistic"]})
        rows.append({"metric": "dm_p_value", "value": dm["p_value"]})

        coefs = results["model2"].get("coefs", {})
        for feat, val in coefs.items():
            if not feat.startswith("cat_"):
                rows.append({"metric": f"coef_{feat}", "value": val})

        return pd.DataFrame(rows)
