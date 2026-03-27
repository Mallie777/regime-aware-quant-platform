from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtester import BacktestResult


def create_tearsheet(result: BacktestResult, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved.append(_plot_equity_and_drawdown(result, output / "equity_and_drawdown.png"))
    saved.append(_plot_exposures(result, output / "exposures.png"))
    saved.append(_plot_monthly_heatmap(result, output / "monthly_returns.png"))
    return saved


def _plot_equity_and_drawdown(result: BacktestResult, path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    result.equity_curve.plot(ax=axes[0], title="Equity Curve")
    axes[0].set_ylabel("Portfolio Value")
    result.risk_report.drawdowns.plot(ax=axes[1], title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0.0, linewidth=1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_exposures(result: BacktestResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    result.exposures[["gross_exposure", "net_exposure", "rolling_beta"]].plot(ax=ax)
    ax.set_title("Portfolio Exposures")
    ax.set_ylabel("Exposure")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_monthly_heatmap(result: BacktestResult, path: Path) -> Path:
    monthly = result.risk_report.monthly_returns.copy()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return").sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto")
    ax.set_title("Monthly Returns Heatmap")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summary_table(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(result.risk_report.summary, index=["value"]).T.rename(columns={"value": "metric"})
