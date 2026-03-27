from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import annualize_return, annualize_volatility, drawdown_series, max_drawdown, sharpe_ratio, sortino_ratio


@dataclass
class RiskReport:
    summary: dict[str, float]
    rolling_metrics: pd.DataFrame
    drawdowns: pd.Series
    monthly_returns: pd.Series


class RiskAnalyzer:
    def summarize(
        self,
        strategy_returns: pd.Series,
        equity_curve: pd.Series,
        annualization: int,
        risk_free_rate: float,
    ) -> RiskReport:
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        ann_return = annualize_return(total_return, len(strategy_returns), annualization)
        ann_vol = annualize_volatility(strategy_returns, annualization)
        sr = sharpe_ratio(strategy_returns, risk_free_rate, annualization)
        sor = sortino_ratio(strategy_returns, risk_free_rate, annualization)
        mdd = max_drawdown(equity_curve)
        calmar = ann_return / abs(mdd) if mdd != 0 else 0.0
        hit_rate = float((strategy_returns > 0).mean())
        avg_win = float(strategy_returns[strategy_returns > 0].mean()) if (strategy_returns > 0).any() else 0.0
        avg_loss = float(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).any() else 0.0
        profit_factor = abs(strategy_returns[strategy_returns > 0].sum() / strategy_returns[strategy_returns < 0].sum()) if (strategy_returns < 0).any() else np.inf
        rolling = pd.DataFrame(
            {
                "rolling_sharpe": strategy_returns.rolling(63).apply(lambda x: sharpe_ratio(pd.Series(x), risk_free_rate, annualization), raw=False),
                "rolling_vol": strategy_returns.rolling(63).std(ddof=1) * np.sqrt(annualization),
                "rolling_return": strategy_returns.rolling(63).apply(lambda x: (1.0 + pd.Series(x)).prod() - 1.0, raw=False),
            }
        )
        dds = drawdown_series(equity_curve)
        monthly = (1.0 + strategy_returns).resample("ME").prod() - 1.0
        summary = {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sr,
            "sortino": sor,
            "max_drawdown": mdd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": float(profit_factor),
        }
        return RiskReport(summary=summary, rolling_metrics=rolling, drawdowns=dds, monthly_returns=monthly)

    def factor_exposures(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:
        exposures = pd.DataFrame(index=weights.index)
        exposures["net_exposure"] = weights.sum(axis=1)
        exposures["gross_exposure"] = weights.abs().sum(axis=1)
        exposures["long_exposure"] = weights.clip(lower=0).sum(axis=1)
        exposures["short_exposure"] = weights.clip(upper=0).abs().sum(axis=1)
        rolling_beta = []
        for date in weights.index:
            hist_r = returns.loc[:date].tail(60)
            hist_b = benchmark_returns.loc[:date].tail(60)
            port_hist = hist_r.mul(weights.loc[date], axis=1).sum(axis=1)
            aligned = pd.concat([port_hist, hist_b], axis=1).dropna()
            if len(aligned) < 10 or aligned.iloc[:, 1].var() == 0:
                rolling_beta.append(0.0)
            else:
                beta = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1] / np.var(aligned.iloc[:, 1], ddof=1)
                rolling_beta.append(float(beta))
        exposures["rolling_beta"] = rolling_beta
        return exposures
