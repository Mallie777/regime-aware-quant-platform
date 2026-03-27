from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SignalConfig


@dataclass
class RegimeOutput:
    features: pd.DataFrame
    scores: pd.DataFrame
    labels: pd.Series
    risk_multiplier: pd.Series


class RegimeDetector:
    """Heuristic regime detector using internal market statistics.

    It classifies each day into one of:
    - risk_on
    - neutral
    - stress
    """

    def __init__(self, config: SignalConfig):
        self.config = config

    def detect(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> RegimeOutput:
        feature_frame = self._build_regime_features(returns, benchmark_returns)
        scores = self._score_regimes(feature_frame)
        labels = scores.idxmax(axis=1)
        risk_multiplier = labels.map({"risk_on": 1.15, "neutral": 1.0, "stress": 0.55}).astype(float)
        return RegimeOutput(features=feature_frame, scores=scores, labels=labels, risk_multiplier=risk_multiplier)

    def _build_regime_features(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
        market_vol = benchmark_returns.rolling(self.config.vol_window).std(ddof=0) * np.sqrt(252)
        cross_dispersion = returns.std(axis=1)
        avg_corr = self._average_pairwise_correlation(returns, window=20)
        breadth = (returns > 0).mean(axis=1)
        momentum = benchmark_returns.rolling(self.config.long_mom_window).sum()
        tail_risk = returns.quantile(0.05, axis=1)
        crashiness = (returns < -0.03).mean(axis=1)
        frame = pd.DataFrame(
            {
                "market_vol": market_vol,
                "cross_dispersion": cross_dispersion,
                "avg_corr": avg_corr,
                "breadth": breadth,
                "benchmark_momentum": momentum,
                "tail_risk": tail_risk,
                "crashiness": crashiness,
            }
        )
        return frame

    def _score_regimes(self, features: pd.DataFrame) -> pd.DataFrame:
        z = (features - features.rolling(120).mean()) / features.rolling(120).std(ddof=0)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scores = pd.DataFrame(index=features.index)
        scores["risk_on"] = (
            0.35 * z["benchmark_momentum"]
            + 0.20 * z["breadth"]
            - 0.20 * z["market_vol"]
            - 0.10 * z["avg_corr"]
            - 0.15 * z["crashiness"]
        )
        scores["stress"] = (
            0.30 * z["market_vol"]
            + 0.20 * z["avg_corr"]
            + 0.20 * (-z["tail_risk"])
            + 0.15 * z["cross_dispersion"]
            + 0.15 * z["crashiness"]
        )
        scores["neutral"] = -scores[["risk_on", "stress"]].abs().mean(axis=1)
        return scores

    def _average_pairwise_correlation(self, returns: pd.DataFrame, window: int) -> pd.Series:
        vals = []
        for i in range(len(returns)):
            if i < window - 1:
                vals.append(np.nan)
                continue
            window_slice = returns.iloc[i - window + 1 : i + 1]
            corr = window_slice.corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            vals.append(float(upper.stack().mean()))
        return pd.Series(vals, index=returns.index)
