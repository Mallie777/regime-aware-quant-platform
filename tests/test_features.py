from __future__ import annotations

import pandas as pd

from quant_showcase.config import SignalConfig
from quant_showcase.data import SyntheticMarketGenerator
from quant_showcase.features import FeatureEngineer


def test_feature_bundle_shapes() -> None:
    market = SyntheticMarketGenerator(config=type("Cfg", (), {
        "n_assets": 8,
        "n_days": 180,
        "start_price": 100.0,
        "seed": 4,
        "benchmark_symbol": "MKT",
        "sectors": ("Tech", "Financials", "Healthcare"),
    })()).generate()
    engineer = FeatureEngineer(SignalConfig())
    bundle = engineer.build_features(
        prices=market.prices,
        returns=market.returns,
        volumes=market.volumes,
        benchmark_returns=market.benchmark_returns,
        fundamentals=market.fundamentals,
    )
    assert bundle.composite.shape == market.prices.shape
    assert set(bundle.normalized.keys())
    assert bundle.composite.index.equals(market.prices.index)
    assert bundle.composite.columns.equals(market.prices.columns)


def test_normalized_features_center_around_zero() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    columns = [f"A{i}" for i in range(6)]
    raw = pd.DataFrame({c: range(i, i + len(index)) for i, c in enumerate(columns)}, index=index, dtype=float)
    normalized = FeatureEngineer(SignalConfig())._normalize_feature(raw)
    daily_means = normalized.mean(axis=1).dropna()
    assert (daily_means.abs() < 1e-6).mean() > 0.9
