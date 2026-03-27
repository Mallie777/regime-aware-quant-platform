from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class SyntheticMarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    volumes: pd.DataFrame
    benchmark: pd.Series
    benchmark_returns: pd.Series
    sectors: dict[str, str]
    regime_states: pd.Series
    fundamentals: pd.DataFrame


class SyntheticMarketGenerator:
    """Generate a realistic synthetic multi-asset universe with hidden regimes.

    The goal is not to mimic exact market microstructure, but to create data with:
    - market-wide shocks
    - sector co-movement
    - idiosyncratic noise
    - changing drift/volatility regimes
    - evolving liquidity and fundamentals
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> SyntheticMarketData:
        dates = pd.bdate_range("2018-01-01", periods=self.config.n_days)
        symbols = [f"AST{i:02d}" for i in range(self.config.n_assets)]
        sectors = self._assign_sectors(symbols)
        regime_states = self._generate_regime_chain(dates)
        benchmark_returns = self._generate_benchmark_returns(regime_states)
        sector_returns = self._generate_sector_returns(dates, regime_states, sectors)
        returns = self._generate_asset_returns(symbols, dates, sectors, regime_states, benchmark_returns, sector_returns)
        prices = self._returns_to_prices(returns, self.config.start_price)
        benchmark = self._returns_to_price_series(benchmark_returns, self.config.start_price)
        volumes = self._generate_volumes(returns, regime_states)
        fundamentals = self._generate_fundamentals(symbols, dates, sectors, regime_states)
        return SyntheticMarketData(
            prices=prices,
            returns=returns,
            volumes=volumes,
            benchmark=benchmark.rename(self.config.benchmark_symbol),
            benchmark_returns=benchmark_returns.rename(self.config.benchmark_symbol),
            sectors=sectors,
            regime_states=regime_states,
            fundamentals=fundamentals,
        )

    def _assign_sectors(self, symbols: list[str]) -> dict[str, str]:
        sector_map: dict[str, str] = {}
        for i, symbol in enumerate(symbols):
            sector_map[symbol] = self.config.sectors[i % len(self.config.sectors)]
        return sector_map

    def _generate_regime_chain(self, dates: pd.DatetimeIndex) -> pd.Series:
        transition = np.array(
            [
                [0.90, 0.08, 0.02],
                [0.08, 0.84, 0.08],
                [0.04, 0.14, 0.82],
            ]
        )
        # 0 = bull, 1 = choppy, 2 = stress
        state = 0
        states = []
        for _ in dates:
            states.append(state)
            probs = transition[state]
            state = int(self.rng.choice([0, 1, 2], p=probs))
        return pd.Series(states, index=dates, name="regime")

    def _generate_benchmark_returns(self, regimes: pd.Series) -> pd.Series:
        mu = {0: 0.00055, 1: 0.00010, 2: -0.00045}
        sigma = {0: 0.008, 1: 0.012, 2: 0.020}
        jumps = {0: 0.001, 1: 0.004, 2: 0.015}
        vals = []
        prev = 0.0
        for regime in regimes.values:
            shock = self.rng.normal(mu[regime], sigma[regime])
            jump = self.rng.normal(0, jumps[regime]) if self.rng.random() < 0.08 else 0.0
            ret = 0.15 * prev + shock + jump
            vals.append(ret)
            prev = ret
        return pd.Series(vals, index=regimes.index, name=self.config.benchmark_symbol)

    def _generate_sector_returns(
        self,
        dates: pd.DatetimeIndex,
        regimes: pd.Series,
        sectors: dict[str, str],
    ) -> pd.DataFrame:
        unique_sectors = sorted(set(sectors.values()))
        sector_df = pd.DataFrame(index=dates, columns=unique_sectors, dtype=float)
        for sector in unique_sectors:
            base_alpha = self.rng.normal(0.0, 0.00012)
            prev = 0.0
            vals = []
            for regime in regimes.values:
                cyclical_boost = 0.00020 if sector in {"Tech", "Consumer"} and regime == 0 else 0.0
                defensive_boost = 0.00018 if sector == "Healthcare" and regime == 2 else 0.0
                energy_spike = 0.00022 if sector == "Energy" and regime != 0 else 0.0
                sigma = 0.006 + 0.002 * regime
                innovation = self.rng.normal(base_alpha + cyclical_boost + defensive_boost + energy_spike, sigma)
                value = 0.25 * prev + innovation
                vals.append(value)
                prev = value
            sector_df[sector] = vals
        return sector_df

    def _generate_asset_returns(
        self,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        sectors: dict[str, str],
        regimes: pd.Series,
        benchmark_returns: pd.Series,
        sector_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        returns = pd.DataFrame(index=dates, columns=symbols, dtype=float)
        style_betas = self.rng.normal(0, 1, size=(len(symbols), 3))
        style_factor_names = ["value", "growth", "quality"]
        style_factor_returns = self._generate_style_factors(dates, regimes, style_factor_names)

        for idx, symbol in enumerate(symbols):
            sector = sectors[symbol]
            market_beta = np.clip(self.rng.normal(1.0, 0.2), 0.5, 1.6)
            sector_beta = np.clip(self.rng.normal(0.9, 0.2), 0.3, 1.4)
            liquidity_profile = np.clip(self.rng.normal(1.0, 0.25), 0.4, 1.8)
            mean_reversion_strength = np.clip(self.rng.normal(-0.18, 0.07), -0.35, -0.05)
            stock_specific_alpha = self.rng.normal(0.00002, 0.00008)
            prev_eps = 0.0
            prev_ret = 0.0
            series = []
            for t, date in enumerate(dates):
                regime = int(regimes.iloc[t])
                market_component = market_beta * benchmark_returns.iloc[t]
                sector_component = sector_beta * sector_returns.loc[date, sector]
                style_component = float(np.dot(style_betas[idx], style_factor_returns.loc[date].values))
                regime_noise_scale = {0: 0.008, 1: 0.012, 2: 0.022}[regime]
                liquidity_drag = 0.00015 * max(liquidity_profile - 1.0, 0) * (1 + regime)
                reversal_component = mean_reversion_strength * prev_ret
                earnings_drift = 0.10 * prev_eps
                eps_surprise = self.rng.normal(0.0, 0.0016 if regime < 2 else 0.0025)
                idio = self.rng.normal(0, regime_noise_scale)
                ret = (
                    stock_specific_alpha
                    + market_component
                    + 0.65 * sector_component
                    + 0.22 * style_component
                    + reversal_component
                    + earnings_drift
                    + eps_surprise
                    + idio
                    - liquidity_drag
                )
                ret = float(np.clip(ret, -0.18, 0.18))
                series.append(ret)
                prev_ret = ret
                prev_eps = eps_surprise
            returns[symbol] = series
        return returns

    def _generate_style_factors(
        self,
        dates: pd.DatetimeIndex,
        regimes: pd.Series,
        factor_names: list[str],
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=dates, columns=factor_names, dtype=float)
        for factor in factor_names:
            prev = 0.0
            vals = []
            for regime in regimes.values:
                mu = 0.00008 if factor == "quality" else 0.0
                if factor == "growth" and regime == 0:
                    mu += 0.00010
                if factor == "value" and regime == 2:
                    mu += 0.00010
                sigma = 0.003 + 0.001 * regime
                value = 0.20 * prev + self.rng.normal(mu, sigma)
                vals.append(value)
                prev = value
            df[factor] = vals
        return df

    def _returns_to_prices(self, returns: pd.DataFrame, start_price: float) -> pd.DataFrame:
        return start_price * (1.0 + returns).cumprod()

    def _returns_to_price_series(self, returns: pd.Series, start_price: float) -> pd.Series:
        return start_price * (1.0 + returns).cumprod()

    def _generate_volumes(self, returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        base = self.rng.lognormal(mean=13.0, sigma=0.35, size=returns.shape[1])
        vols = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for col_idx, col in enumerate(returns.columns):
            prev = base[col_idx]
            series = []
            for i, date in enumerate(returns.index):
                regime = int(regimes.loc[date])
                shock_multiplier = 1.0 + 10.0 * abs(returns.iloc[i, col_idx])
                regime_multiplier = {0: 0.95, 1: 1.10, 2: 1.35}[regime]
                innovation = self.rng.lognormal(mean=0.0, sigma=0.18)
                volume = 0.72 * prev + 0.28 * base[col_idx] * innovation * shock_multiplier * regime_multiplier
                series.append(volume)
                prev = volume
            vols[col] = series
        return vols

    def _generate_fundamentals(
        self,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        sectors: dict[str, str],
        regimes: pd.Series,
    ) -> pd.DataFrame:
        records = []
        quarterly_dates = dates[::63]
        for symbol in symbols:
            sector = sectors[symbol]
            valuation_anchor = self.rng.normal(18, 4)
            leverage_anchor = self.rng.normal(0.45, 0.12)
            profit_anchor = self.rng.normal(0.10, 0.03)
            for date in quarterly_dates:
                regime = int(regimes.loc[date])
                pe = valuation_anchor + self.rng.normal(0, 1.0) + (1.2 if sector == "Tech" else 0.0)
                debt_to_equity = max(0.0, leverage_anchor + self.rng.normal(0, 0.05) + (0.08 if regime == 2 else 0.0))
                roe = profit_anchor + self.rng.normal(0, 0.015) - (0.015 if regime == 2 else 0.0)
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "sector": sector,
                        "pe_ratio": float(max(pe, 4.0)),
                        "debt_to_equity": float(debt_to_equity),
                        "roe": float(roe),
                    }
                )
        fundamentals = pd.DataFrame.from_records(records)
        return fundamentals
