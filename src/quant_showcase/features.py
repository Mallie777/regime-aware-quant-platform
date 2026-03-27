from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SignalConfig
from .utils import cross_sectional_zscore, pct_rank, rolling_zscore, winsorize_cross_section


@dataclass
class FeatureBundle:
    raw: dict[str, pd.DataFrame]
    normalized: dict[str, pd.DataFrame]
    composite: pd.DataFrame


class FeatureEngineer:
    def __init__(self, config: SignalConfig):
        self.config = config

    def build_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        benchmark_returns: pd.Series,
        fundamentals: pd.DataFrame | None = None,
    ) -> FeatureBundle:
        raw = {
            "short_term_reversal": self.short_term_reversal(prices),
            "medium_term_momentum": self.medium_term_momentum(prices),
            "volatility_adjusted_momentum": self.volatility_adjusted_momentum(prices, returns),
            "breadth_relative_strength": self.breadth_relative_strength(returns, benchmark_returns),
            "liquidity_stability": self.liquidity_stability(volumes),
            "quality_of_trend": self.quality_of_trend(prices),
        }

        if fundamentals is not None and not fundamentals.empty:
            raw["fundamental_quality"] = self.fundamental_quality(fundamentals, prices.index, prices.columns)

        normalized: dict[str, pd.DataFrame] = {}
        for name, frame in raw.items():
            normalized[name] = self._normalize_feature(frame)

        composite = self._combine_features(normalized)
        return FeatureBundle(raw=raw, normalized=normalized, composite=composite)

    def short_term_reversal(self, prices: pd.DataFrame) -> pd.DataFrame:
        lookback = self.config.short_mr_window
        recent = prices.pct_change(lookback)
        return -recent

    def medium_term_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        window = self.config.long_mom_window
        return prices.pct_change(window)

    def volatility_adjusted_momentum(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        mom = prices.pct_change(self.config.long_mom_window)
        vol = returns.rolling(self.config.vol_window).std(ddof=0) * np.sqrt(252)
        return mom / vol.replace(0, np.nan)

    def breadth_relative_strength(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
        advancers = (returns > 0).rolling(self.config.breadth_window).mean()
        benchmark_trend = benchmark_returns.rolling(self.config.breadth_window).mean()
        relative = advancers.sub(benchmark_trend, axis=0)
        return relative

    def liquidity_stability(self, volumes: pd.DataFrame) -> pd.DataFrame:
        avg = volumes.rolling(20).mean()
        std = volumes.rolling(20).std(ddof=0)
        coeff_var = std / avg.replace(0, np.nan)
        stable_liquidity = -coeff_var
        return stable_liquidity

    def quality_of_trend(self, prices: pd.DataFrame) -> pd.DataFrame:
        slope = prices.pct_change(self.config.quality_window)
        path = prices.pct_change().abs().rolling(self.config.quality_window).sum()
        return slope / path.replace(0, np.nan)

    def fundamental_quality(
        self,
        fundamentals: pd.DataFrame,
        dates: pd.Index,
        symbols: pd.Index,
    ) -> pd.DataFrame:
        pivot_roe = fundamentals.pivot(index="date", columns="symbol", values="roe").reindex(dates).ffill()
        pivot_dte = fundamentals.pivot(index="date", columns="symbol", values="debt_to_equity").reindex(dates).ffill()
        pivot_pe = fundamentals.pivot(index="date", columns="symbol", values="pe_ratio").reindex(dates).ffill()
        score = pivot_roe - 0.5 * pivot_dte - 0.03 * pivot_pe
        return score.reindex(columns=symbols)

    def _normalize_feature(self, feature: pd.DataFrame) -> pd.DataFrame:
        cleaned = feature.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.ffill(limit=5)
        cleaned = winsorize_cross_section(cleaned, self.config.winsor_quantile)
        z = cross_sectional_zscore(cleaned, clip=self.config.zscore_clip)
        return z

    def _combine_features(self, normalized: dict[str, pd.DataFrame]) -> pd.DataFrame:
        weighted_sum = None
        total_weight = 0.0
        for name, frame in normalized.items():
            weight = self.config.feature_weights.get(name, 0.08)
            total_weight += weight
            weighted_component = frame * weight
            weighted_sum = weighted_component if weighted_sum is None else weighted_sum.add(weighted_component, fill_value=0.0)
        if weighted_sum is None or total_weight == 0:
            raise ValueError("No features available to combine")
        combined = weighted_sum / total_weight
        smoothed = combined.ewm(alpha=self.config.signal_smoothing).mean()
        return smoothed

    def feature_importance_snapshot(self, bundle: FeatureBundle) -> pd.DataFrame:
        rows = []
        for name, frame in bundle.normalized.items():
            rows.append(
                {
                    "feature": name,
                    "mean_abs_score": float(frame.abs().stack().mean()),
                    "cross_sectional_dispersion": float(frame.std(axis=1).mean()),
                    "coverage": float(frame.notna().mean().mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("mean_abs_score", ascending=False)

    def diagnostic_panel(self, bundle: FeatureBundle) -> dict[str, pd.DataFrame]:
        diagnostics: dict[str, pd.DataFrame] = {}
        for name, frame in bundle.normalized.items():
            diagnostics[name] = pd.DataFrame(
                {
                    "daily_mean": frame.mean(axis=1),
                    "daily_std": frame.std(axis=1),
                    "daily_skew_proxy": (frame.pow(3).mean(axis=1)).clip(-10, 10),
                    "top_decile_share": (pct_rank(frame) >= 0.9).mean(axis=1),
                }
            )
        return diagnostics

    def rolling_alpha_decay(self, feature: pd.DataFrame, forward_returns: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        rows = []
        for horizon in horizons:
            fwd = forward_returns.shift(-horizon)
            ic_series = []
            for date in feature.index:
                x = feature.loc[date]
                y = fwd.loc[date]
                aligned = pd.concat([x, y], axis=1).dropna()
                if len(aligned) < 5:
                    ic_series.append(np.nan)
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
                ic_series.append(corr)
            rows.append({"horizon": horizon, "mean_rank_ic": float(pd.Series(ic_series).mean())})
        return pd.DataFrame(rows)

    def residual_momentum(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
        residuals = returns.sub(benchmark_returns, axis=0)
        return residuals.rolling(self.config.long_mom_window).sum()

    def volatility_regime_score(self, returns: pd.DataFrame) -> pd.DataFrame:
        realized = returns.rolling(self.config.vol_window).std(ddof=0)
        return rolling_zscore(realized, self.config.vol_window)
