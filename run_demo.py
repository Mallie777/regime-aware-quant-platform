from __future__ import annotations

from pathlib import Path

from quant_showcase.backtester import Backtester
from quant_showcase.config import BacktestConfig, DataConfig, PortfolioConfig, SignalConfig
from quant_showcase.data import SyntheticMarketGenerator
from quant_showcase.features import FeatureEngineer
from quant_showcase.portfolio import PortfolioConstructor
from quant_showcase.regimes import RegimeDetector
from quant_showcase.reporting import create_tearsheet, summary_table
from quant_showcase.signals import SignalEngine


def main() -> None:
    data_cfg = DataConfig(n_assets=10, n_days=200, seed=42)
    signal_cfg = SignalConfig()
    portfolio_cfg = PortfolioConfig()
    backtest_cfg = BacktestConfig()

    market = SyntheticMarketGenerator(data_cfg).generate()
    features = FeatureEngineer(signal_cfg).build_features(
        prices=market.prices,
        returns=market.returns,
        volumes=market.volumes,
        benchmark_returns=market.benchmark_returns,
        fundamentals=market.fundamentals,
    )
    regimes = RegimeDetector(signal_cfg).detect(market.returns, market.benchmark_returns)
    signals = SignalEngine(signal_cfg).generate(features, regimes, market.returns)
    portfolio = PortfolioConstructor(portfolio_cfg).construct(
        signal=signals.regime_adjusted_signal,
        returns=market.returns,
        sectors=market.sectors,
        benchmark_returns=market.benchmark_returns,
        risk_multiplier=regimes.risk_multiplier,
    )
    result = Backtester(backtest_cfg).run(portfolio, market.returns, market.benchmark_returns)

    print("\n=== PERFORMANCE SUMMARY ===")
    print(summary_table(result).round(4).to_string())
    print("\n=== LAST 5 EXPOSURES ===")
    print(result.exposures.tail().round(4).to_string())
    print("\n=== TRADE LOG ===")
    print(result.trade_log.tail(10).to_string(index=False) if not result.trade_log.empty else "No drawdown stop events triggered")

    output_dir = Path("outputs")
    saved = create_tearsheet(result, output_dir)
    print("\nSaved charts:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
