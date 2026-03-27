from __future__ import annotations

import numpy as np
import pandas as pd

from quant_showcase.backtester import Backtester
from quant_showcase.config import BacktestConfig
from quant_showcase.portfolio import PortfolioState


def test_backtester_produces_consistent_shapes() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(123)
    weights = pd.DataFrame(rng.normal(0, 0.1, size=(120, 4)), index=dates, columns=cols)
    turnover = pd.Series(rng.uniform(0, 0.4, size=120), index=dates)
    portfolio = PortfolioState(
        weights=weights,
        gross_exposure=weights.abs().sum(axis=1),
        net_exposure=weights.sum(axis=1),
        turnover=turnover,
        beta_exposure=pd.Series(0.0, index=dates),
        leverage_scaler=pd.Series(1.0, index=dates),
    )
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(120, 4)), index=dates, columns=cols)
    benchmark = pd.Series(rng.normal(0, 0.008, size=120), index=dates)
    result = Backtester(BacktestConfig()).run(portfolio, returns, benchmark)
    assert result.daily_returns.index.equals(dates)
    assert result.equity_curve.index.equals(dates)
    assert len(result.costs) == len(dates)
    assert not result.exposures.empty
