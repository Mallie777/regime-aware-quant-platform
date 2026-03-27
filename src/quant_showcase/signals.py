from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SignalConfig
from .features import FeatureBundle
from .regimes import RegimeOutput
from .utils import cross_sectional_zscore


@dataclass
class SignalOutput:
    base_signal: pd.DataFrame
    regime_adjusted_signal: pd.DataFrame
    conviction: pd.DataFrame
    signal_quality: pd.Series


class SignalEngine:
    def __init__(self, config: SignalConfig):
        self.config = config

    def generate(
        self,
        features: FeatureBundle,
        regimes: RegimeOutput,
        returns: pd.DataFrame,
    ) -> SignalOutput:
        base = features.composite.copy()
        conviction = self._estimate_conviction(features.normalized)
        adjusted = self._apply_regime_logic(base, features.normalized, regimes, conviction)
        adjusted = self._stabilize(adjusted)
        signal_quality = self._compute_signal_quality(adjusted, returns)
        return SignalOutput(
            base_signal=base,
            regime_adjusted_signal=adjusted,
            conviction=conviction,
            signal_quality=signal_quality,
        )

    def _estimate_conviction(self, normalized_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        feature_stack = list(normalized_features.values())
        if not feature_stack:
            raise ValueError("No normalized features available")
        dispersion = sum(frame.abs() for frame in feature_stack) / len(feature_stack)
        agreement = self._feature_agreement(normalized_features)
        conviction = 0.65 * dispersion + 0.35 * agreement
        return conviction.clip(lower=0.0)

    def _feature_agreement(self, normalized_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        names = list(normalized_features.keys())
        base = None
        count = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = np.sign(normalized_features[names[i]])
                b = np.sign(normalized_features[names[j]])
                pair = (a == b).astype(float)
                base = pair if base is None else base.add(pair, fill_value=0.0)
                count += 1
        if base is None or count == 0:
            first = normalized_features[names[0]]
            return pd.DataFrame(1.0, index=first.index, columns=first.columns)
        return base / count

    def _apply_regime_logic(
        self,
        base_signal: pd.DataFrame,
        normalized_features: dict[str, pd.DataFrame],
        regimes: RegimeOutput,
        conviction: pd.DataFrame,
    ) -> pd.DataFrame:
        adjusted = base_signal.copy()
        momentum = normalized_features.get("medium_term_momentum", base_signal)
        reversal = normalized_features.get("short_term_reversal", -base_signal)
        vol_adj_mom = normalized_features.get("volatility_adjusted_momentum", momentum)

        for date in adjusted.index:
            regime = regimes.labels.loc[date]
            rm = regimes.risk_multiplier.loc[date]
            if regime == "risk_on":
                combined = 0.45 * momentum.loc[date] + 0.25 * vol_adj_mom.loc[date] + 0.30 * adjusted.loc[date]
            elif regime == "stress":
                combined = 0.42 * reversal.loc[date] + 0.25 * adjusted.loc[date] + 0.33 * (-momentum.loc[date])
            else:
                combined = 0.55 * adjusted.loc[date] + 0.20 * momentum.loc[date] + 0.25 * reversal.loc[date]
            combined = combined * conviction.loc[date] * rm * self.config.regime_sensitivity
            adjusted.loc[date] = combined
        return cross_sectional_zscore(adjusted, clip=self.config.zscore_clip)

    def _stabilize(self, signal: pd.DataFrame) -> pd.DataFrame:
        ewm = signal.ewm(alpha=self.config.signal_smoothing).mean()
        stale_mask = signal.notna().rolling(3).sum().fillna(0) < 2
        ewm[stale_mask] = np.nan
        return ewm

    def _compute_signal_quality(self, signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        fwd = returns.shift(-1)
        ics = []
        for date in signal.index:
            x = signal.loc[date]
            y = fwd.loc[date]
            aligned = pd.concat([x, y], axis=1).dropna()
            if len(aligned) < 5:
                ics.append(np.nan)
            else:
                ics.append(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))
        return pd.Series(ics, index=signal.index, name="daily_rank_ic").rolling(20).mean()
