from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg") 

import matplotlib.pyplot as plt

from src.backtest.result import BacktestResult

DEFAULT_REPORTS_DIR = Path("reports")


class ReportPlotter:
    def __init__(self, output_dir: Path | str = DEFAULT_REPORTS_DIR) -> None:
        self.output_dir = Path(output_dir)

    def plot(self, result: BacktestResult, benchmark: BacktestResult, filename: str = "report.png") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(12, 8))

        self._plot_price_and_signals(ax_price, result)
        self._plot_equity_curves(ax_equity, result, benchmark)

        fig.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def _plot_price_and_signals(self, ax, result: BacktestResult) -> None:
        df = result.df
        ax.plot(df.index, df["close"], label="Close", color="black", linewidth=1)

        for column in df.columns:
            if column.startswith("sma_") or column.startswith("ema_"):
                ax.plot(df.index, df[column], label=column, linewidth=1)

        position_change = result.positions.diff().fillna(result.positions.iloc[0])
        buys = position_change[position_change > 0].index
        sells = position_change[position_change < 0].index
        ax.scatter(buys, df.loc[buys, "close"], marker="^", color="green", label="Buy", zorder=5)
        ax.scatter(sells, df.loc[sells, "close"], marker="v", color="red", label="Sell", zorder=5)

        ax.set_title(f"{result.strategy_name} - Price & Signals")
        ax.legend(loc="best")

    def _plot_equity_curves(self, ax, result: BacktestResult, benchmark: BacktestResult) -> None:
        ax.plot(result.equity_curve.index, result.equity_curve, label=result.strategy_name)
        ax.plot(benchmark.equity_curve.index, benchmark.equity_curve, label=benchmark.strategy_name)
        ax.set_title("Equity Curve: Strategy vs Benchmark")
        ax.legend(loc="best")
