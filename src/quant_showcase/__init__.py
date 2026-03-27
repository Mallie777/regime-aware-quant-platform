"""Quant showcase package."""

from .config import BacktestConfig, DataConfig, PortfolioConfig, SignalConfig
from .data import SyntheticMarketGenerator
from .features import FeatureEngineer
from .regimes import RegimeDetector
from .signals import SignalEngine
from .portfolio import PortfolioConstructor
from .backtester import Backtester
from .reporting import create_tearsheet

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "PortfolioConfig",
    "SignalConfig",
    "SyntheticMarketGenerator",
    "FeatureEngineer",
    "RegimeDetector",
    "SignalEngine",
    "PortfolioConstructor",
    "Backtester",
    "create_tearsheet",
]
