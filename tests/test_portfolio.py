from __future__ import annotations

import numpy as np
import pandas as pd

from quant_showcase.config import PortfolioConfig
from quant_showcase.portfolio import PortfolioConstructor


def test_constructed_weights_respect_gross_limits() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    cols = [f"A{i}" for i in range(10)]
    rng = np.random.default_rng(10)
    signal = pd.DataFrame(rng.normal(size=(80, 10)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(80, 10)), index=dates, columns=cols)
    benchmark = pd.Series(rng.normal(0, 0.008, size=80), index=dates)
    sectors = {c: ["Tech", "Health", "Fin"][i % 3] for i, c in enumerate(cols)}
    risk_multiplier = pd.Series(1.0, index=dates)
    portfolio = PortfolioConstructor(PortfolioConfig()).construct(signal, returns, sectors, benchmark, risk_multiplier)
    assert (portfolio.gross_exposure <= PortfolioConfig().gross_target + 1e-6).all()


def test_net_exposure_close_to_zero_for_market_neutral_setup() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    cols = [f"A{i}" for i in range(10)]
    rng = np.random.default_rng(11)
    signal = pd.DataFrame(rng.normal(size=(80, 10)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(80, 10)), index=dates, columns=cols)
    benchmark = pd.Series(rng.normal(0, 0.008, size=80), index=dates)
    sectors = {c: ["Tech", "Health", "Fin"][i % 3] for i, c in enumerate(cols)}
    risk_multiplier = pd.Series(1.0, index=dates)
    portfolio = PortfolioConstructor(PortfolioConfig()).construct(signal, returns, sectors, benchmark, risk_multiplier)
    assert float(portfolio.net_exposure.abs().mean()) < 0.05
