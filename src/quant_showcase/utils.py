from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def to_frame(data: pd.Series | pd.DataFrame, name: str | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    out = data.to_frame(name=name or data.name or "value")
    return out


def annualize_return(period_return: float, periods: int, annualization: int = TRADING_DAYS) -> float:
    if periods <= 0:
        return 0.0
    return (1.0 + period_return) ** (annualization / periods) - 1.0


def annualize_volatility(returns: pd.Series, annualization: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(annualization))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = TRADING_DAYS,
) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    rf_daily = risk_free_rate / annualization
    excess = returns - rf_daily
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(np.sqrt(annualization) * excess.mean() / denom)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = TRADING_DAYS,
) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    rf_daily = risk_free_rate / annualization
    downside = np.minimum(returns - rf_daily, 0.0)
    downside_std = np.sqrt(np.mean(np.square(downside)))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(annualization) * (returns.mean() - rf_daily) / downside_std)


def max_drawdown(equity_curve: pd.Series) -> float:
    curve = equity_curve.dropna()
    if curve.empty:
        return 0.0
    running_max = curve.cummax()
    drawdowns = curve / running_max - 1.0
    return float(drawdowns.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    curve = equity_curve.dropna()
    running_max = curve.cummax()
    return curve / running_max - 1.0


def rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window).mean()
    std = frame.rolling(window).std(ddof=0)
    return (frame - mean) / std.replace(0, np.nan)


def cross_sectional_zscore(frame: pd.DataFrame, clip: float | None = None) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    z = frame.sub(mean, axis=0).div(std, axis=0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def winsorize_cross_section(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    lower = frame.quantile(quantile, axis=1)
    upper = frame.quantile(1.0 - quantile, axis=1)
    clipped = frame.copy()
    for idx in frame.index:
        clipped.loc[idx] = clipped.loc[idx].clip(lower.loc[idx], upper.loc[idx])
    return clipped


def normalize_weights(weights: pd.Series, gross_target: float, max_abs_weight: float) -> pd.Series:
    if weights.abs().sum() == 0:
        return weights.copy()
    out = weights / weights.abs().sum() * gross_target
    out = out.clip(-max_abs_weight, max_abs_weight)
    gross = out.abs().sum()
    if gross == 0:
        return out
    return out / gross * gross_target


def ensure_monotonic_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame


def realized_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    var_bench = cov[1, 1]
    if var_bench == 0:
        return 0.0
    return float(cov[0, 1] / var_bench)


def compute_turnover(new_weights: pd.Series, old_weights: pd.Series | None) -> float:
    if old_weights is None:
        return float(new_weights.abs().sum())
    aligned = pd.concat([new_weights, old_weights], axis=1).fillna(0.0)
    return float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum())


def rolling_apply(df: pd.DataFrame, window: int, func) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        out[col] = df[col].rolling(window).apply(func, raw=False)
    return out


def sigmoid(x: pd.Series | pd.DataFrame | np.ndarray, scale: float = 1.0):
    return 1.0 / (1.0 + np.exp(-scale * x))


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def config_to_dict(*configs: object) -> dict:
    merged: dict[str, object] = {}
    for cfg in configs:
        if is_dataclass(cfg):
            merged[cfg.__class__.__name__] = asdict(cfg)
    return merged


def pct_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def capped_forward_fill(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return frame.ffill(limit=limit)


def validate_equal_index(*objs: Iterable[pd.Index] | pd.DataFrame | pd.Series) -> None:
    indexes: list[pd.Index] = []
    for obj in objs:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            indexes.append(obj.index)
        elif isinstance(obj, pd.Index):
            indexes.append(obj)
    if not indexes:
        return
    first = indexes[0]
    for idx in indexes[1:]:
        if not first.equals(idx):
            raise ValueError("Indexes are not aligned")
