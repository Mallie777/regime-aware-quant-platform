from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BacktestConfig
from .portfolio import PortfolioState
from .risk import RiskAnalyzer, RiskReport


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    costs: pd.Series
    turnover: pd.Series
    exposures: pd.DataFrame
    risk_report: RiskReport
    trade_log: pd.DataFrame


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_analyzer = RiskAnalyzer()

    def run(
        self,
        portfolio: PortfolioState,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> BacktestResult:
        weights = portfolio.weights.shift(1).fillna(0.0)
        raw_strategy_returns = (weights * asset_returns).sum(axis=1)
        costs = self._compute_costs(portfolio.turnover, portfolio.weights)
        gated_returns, trade_log = self._apply_drawdown_control(raw_strategy_returns - costs)
        equity_curve = self.config.initial_capital * (1.0 + gated_returns).cumprod()
        risk_report = self.risk_analyzer.summarize(
            strategy_returns=gated_returns,
            equity_curve=equity_curve,
            annualization=self.config.annualization,
            risk_free_rate=self.config.risk_free_rate,
        )
        exposures = self.risk_analyzer.factor_exposures(portfolio.weights, asset_returns, benchmark_returns)
        return BacktestResult(
            daily_returns=gated_returns,
            equity_curve=equity_curve,
            weights=portfolio.weights,
            costs=costs,
            turnover=portfolio.turnover,
            exposures=exposures,
            risk_report=risk_report,
            trade_log=trade_log,
        )

    def _compute_costs(self, turnover: pd.Series, weights: pd.DataFrame) -> pd.Series:
        tc = turnover * (self.config.transaction_cost_bps / 10_000.0)
        slippage = turnover * (self.config.slippage_bps / 10_000.0)
        borrow = weights.clip(upper=0).abs().sum(axis=1) * (self.config.borrow_cost_bps / 10_000.0) / 252.0
        total = (tc + slippage + borrow).rename("costs")
        return total

    def _apply_drawdown_control(self, returns: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
        enabled = True
        cooldown = 0
        equity = 1.0
        peak = 1.0
        out = []
        logs: list[dict] = []
        for date, ret in returns.items():
            if cooldown > 0:
                cooldown -= 1
                enabled = False
            else:
                enabled = True
            applied_ret = ret if enabled else 0.0
            equity *= 1.0 + applied_ret
            peak = max(peak, equity)
            dd = equity / peak - 1.0
            if dd <= -self.config.stop_drawdown and enabled:
                cooldown = self.config.drawdown_cooldown_days
                enabled = False
                logs.append({"date": date, "event": "drawdown_stop", "drawdown": dd})
            out.append(applied_ret)
        trade_log = pd.DataFrame(logs)
        return pd.Series(out, index=returns.index, name="strategy_returns"), trade_log
