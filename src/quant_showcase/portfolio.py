from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .utils import compute_turnover, normalize_weights, realized_beta


@dataclass
class PortfolioState:
    weights: pd.DataFrame
    gross_exposure: pd.Series
    net_exposure: pd.Series
    turnover: pd.Series
    beta_exposure: pd.Series
    leverage_scaler: pd.Series


class PortfolioConstructor:
    def __init__(self, config: PortfolioConfig):
        self.config = config

    def construct(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        sectors: dict[str, str],
        benchmark_returns: pd.Series,
        risk_multiplier: pd.Series,
    ) -> PortfolioState:
        weights = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
        beta_exposure = pd.Series(index=signal.index, dtype=float)
        leverage_scaler = pd.Series(index=signal.index, dtype=float)
        prev_weights: pd.Series | None = None
        turnover_values = []

        for date in signal.index:
            raw = signal.loc[date].dropna()
            if raw.empty:
                weights.loc[date] = 0.0
                beta_exposure.loc[date] = 0.0
                leverage_scaler.loc[date] = 0.0
                turnover_values.append(0.0)
                prev_weights = pd.Series(0.0, index=signal.columns)
                continue

            current = raw.copy()
            if self.config.sector_neutral:
                current = self._sector_neutralize(current, sectors)

            if self.config.net_target == 0.0:
                current = current - current.mean()
            current = self._sparsify(current)
            current = normalize_weights(current, self.config.gross_target, self.config.max_abs_weight)
            current = self._vol_target_scale(current, returns.loc[:date].iloc[-60:], risk_multiplier.loc[date])

            if self.config.beta_neutral:
                current = self._beta_neutralize(current, returns.loc[:date], benchmark_returns.loc[:date])

            current = self._target_net_exposure(current)
            current = self._enforce_cash_buffer(current)
            current = self._target_net_exposure(current)
            full = pd.Series(0.0, index=signal.columns, dtype=float)
            full.loc[current.index] = current.values

            beta_exposure.loc[date] = self._portfolio_beta(full, returns.loc[:date], benchmark_returns.loc[:date])
            leverage_scaler.loc[date] = full.abs().sum() / max(self.config.gross_target, 1e-9)
            turnover_values.append(compute_turnover(full, prev_weights))
            weights.loc[date] = full
            prev_weights = full

        gross = weights.abs().sum(axis=1)
        net = weights.sum(axis=1)
        turnover = pd.Series(turnover_values, index=signal.index, name="turnover")
        return PortfolioState(
            weights=weights,
            gross_exposure=gross,
            net_exposure=net,
            turnover=turnover,
            beta_exposure=beta_exposure,
            leverage_scaler=leverage_scaler,
        )

    def _sector_neutralize(self, scores: pd.Series, sectors: dict[str, str]) -> pd.Series:
        df = pd.DataFrame({"score": scores, "sector": [sectors[s] for s in scores.index]})
        df["adjusted"] = df["score"] - df.groupby("sector")["score"].transform("mean")
        return df["adjusted"]

    def _sparsify(self, scores: pd.Series) -> pd.Series:
        out = scores.copy()
        out[out.abs() < self.config.min_trade_weight] = 0.0
        if out.abs().sum() == 0:
            return scores
        return out

    def _vol_target_scale(
        self,
        weights: pd.Series,
        trailing_returns: pd.DataFrame,
        risk_multiplier: float,
    ) -> pd.Series:
        aligned = trailing_returns.reindex(columns=weights.index).fillna(0.0)
        if len(aligned) < 20:
            return weights * risk_multiplier
        port = aligned.mul(weights, axis=1).sum(axis=1)
        realized_vol = port.std(ddof=1) * np.sqrt(252)
        if realized_vol == 0 or np.isnan(realized_vol):
            return weights * risk_multiplier
        scaler = min((self.config.annual_vol_target / realized_vol) * risk_multiplier, 1.8)
        return weights * scaler

    def _beta_neutralize(
        self,
        weights: pd.Series,
        asset_returns_history: pd.DataFrame,
        benchmark_returns_history: pd.Series,
    ) -> pd.Series:
        betas = {}
        aligned_assets = asset_returns_history.reindex(columns=weights.index)
        for col in weights.index:
            betas[col] = realized_beta(
                aligned_assets[col].iloc[-self.config.beta_window :],
                benchmark_returns_history.iloc[-self.config.beta_window :],
            )
        beta_vec = pd.Series(betas)
        port_beta = float((weights * beta_vec).sum())
        beta_norm = float(np.square(beta_vec).sum())
        if beta_norm == 0:
            return weights
        hedge = port_beta / beta_norm
        adjusted = weights - hedge * beta_vec
        if adjusted.abs().sum() == 0:
            return weights
        return adjusted / adjusted.abs().sum() * min(weights.abs().sum(), self.config.gross_target)

    def _portfolio_beta(
        self,
        weights: pd.Series,
        asset_returns_history: pd.DataFrame,
        benchmark_returns_history: pd.Series,
    ) -> float:
        betas = []
        for col in weights.index:
            beta = realized_beta(
                asset_returns_history[col].iloc[-self.config.beta_window :],
                benchmark_returns_history.iloc[-self.config.beta_window :],
            )
            betas.append(beta)
        beta_vec = pd.Series(betas, index=weights.index)
        return float((weights * beta_vec).sum())


    def _target_net_exposure(self, weights: pd.Series) -> pd.Series:
        target = self.config.net_target
        if weights.empty:
            return weights
        adjusted = weights - (weights.sum() - target) / len(weights)
        gross = adjusted.abs().sum()
        if gross == 0:
            return weights
        gross_cap = min(gross, self.config.gross_target)
        adjusted = adjusted / gross * gross_cap
        return adjusted.clip(-self.config.max_abs_weight, self.config.max_abs_weight)

    def _enforce_cash_buffer(self, weights: pd.Series) -> pd.Series:
        gross = weights.abs().sum()
        cap = max(0.0, self.config.gross_target - self.config.cash_buffer)
        if gross <= cap or gross == 0:
            return weights
        return weights / gross * cap
