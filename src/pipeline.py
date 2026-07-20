from __future__ import annotations

from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.data.base import DataProvider
from src.evaluation.metrics import PerformanceMetrics
from src.evaluation.plots import ReportPlotter
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy


class Pipeline:

    def __init__(
        self,
        provider: DataProvider,
        cleaner: DataCleaner,
        engineer: FeatureEngineer,
        strategies: list[Strategy],
        backtester: Backtester,
        metrics: PerformanceMetrics,
        plotter: ReportPlotter,
    ) -> None:
        self.provider = provider
        self.cleaner = cleaner
        self.engineer = engineer
        self.strategies = strategies
        self.backtester = backtester
        self.metrics = metrics
        self.plotter = plotter

    def run(self, plot_filename: str = "report.png") -> list[BacktestResult]:
        raw = self.provider.fetch()
        clean = self.cleaner.clean(raw)
        featured = self.engineer.returns(clean)

        results = [self.backtester.run(featured, strategy) for strategy in self.strategies]

        self.plotter.plot(results[0], results[-1], filename=plot_filename)
        self.metrics.compare(results)

        return results
