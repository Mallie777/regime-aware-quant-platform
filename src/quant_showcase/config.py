from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DataConfig:
    n_assets: int = 24
    n_days: int = 1000
    start_price: float = 100.0
    seed: int = 7
    benchmark_symbol: str = "MKT"
    sectors: tuple[str, ...] = (
        "Tech",
        "Financials",
        "Industrials",
        "Healthcare",
        "Energy",
        "Consumer",
    )


@dataclass(slots=True)
class SignalConfig:
    short_mr_window: int = 5
    long_mom_window: int = 63
    vol_window: int = 20
    breadth_window: int = 15
    dispersion_window: int = 20
    quality_window: int = 60
    winsor_quantile: float = 0.03
    zscore_clip: float = 3.5
    signal_smoothing: float = 0.3
    regime_sensitivity: float = 1.0
    feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "short_term_reversal": 0.30,
            "medium_term_momentum": 0.27,
            "volatility_adjusted_momentum": 0.18,
            "breadth_relative_strength": 0.10,
            "liquidity_stability": 0.07,
            "quality_of_trend": 0.08,
        }
    )


@dataclass(slots=True)
class PortfolioConfig:
    gross_target: float = 1.6
    net_target: float = 0.0
    max_abs_weight: float = 0.12
    min_trade_weight: float = 0.002
    turnover_penalty: float = 0.15
    annual_vol_target: float = 0.12
    sector_neutral: bool = True
    beta_neutral: bool = True
    beta_window: int = 60
    cash_buffer: float = 0.02


@dataclass(slots=True)
class BacktestConfig:
    rebalance_every: int = 5
    annualization: int = 252
    transaction_cost_bps: float = 8.0
    borrow_cost_bps: float = 1.0
    slippage_bps: float = 4.0
    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.01
    stop_drawdown: float = 0.16
    drawdown_cooldown_days: int = 20
