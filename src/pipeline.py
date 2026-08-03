from __future__ import annotations

from src.analysis.monte_carlo import MonteCarloAnalyzer
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
        monte_carlo: MonteCarloAnalyzer | None = None,
    ) -> None:
        self.provider = provider
        self.cleaner = cleaner
        self.engineer = engineer
        self.strategies = strategies
        self.backtester = backtester
        self.metrics = metrics
        self.plotter = plotter
        self.monte_carlo = monte_carlo

    def run(self, plot_filename: str = "report.png", monte_carlo_filename: str = "monte_carlo.png") -> list[BacktestResult]:
        raw = self.provider.fetch()
        clean = self.cleaner.clean(raw)
        featured = self.engineer.returns(clean)

        results = [self.backtester.run(featured, strategy) for strategy in self.strategies]

        self.plotter.plot(results[0], results[-1], filename=plot_filename)
        self.metrics.compare(results)

        # Monte Carlo analysis is opt-in 
        if self.monte_carlo is not None:
            primary = results[0]
            mc_result = self.monte_carlo.bootstrap(primary)
            self.monte_carlo.print_bootstrap_report(mc_result, label=primary.strategy_name)
            self.plotter.plot_monte_carlo(
                mc_result, label=primary.strategy_name, filename=monte_carlo_filename
            )

            noise_result = self.monte_carlo.noise_robustness(
                featured, self.strategies[0], self.backtester
            )
            self.monte_carlo.print_noise_report(noise_result, label=primary.strategy_name)

        return results
