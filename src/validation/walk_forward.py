from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.evaluation.metrics import PerformanceMetrics
from src.strategies.base import Strategy


@dataclass
class WalkForwardResult:
    """output of a full walk-forward run across all folds"""

    fold_selections: list[dict]
    fold_metrics: list[dict]
    stitched_result: BacktestResult


class WalkForwardValidator:
 
    def __init__(
        self,
        backtester: Backtester,
        metrics: PerformanceMetrics,
        train_size: int,
        test_size: int,
        expanding: bool = False,
    ) -> None:

        self.backtester = backtester
        self.metrics = metrics
        self.train_size = train_size
        self.test_size = test_size
        self.expanding = expanding

    def _folds(self, n_bars: int):
        """Yield (train_start, train_end, test_start, test_end) bar indices for each fold"""
        test_start = self.train_size
        while test_start + self.test_size <= n_bars:
            train_start = 0 if self.expanding else test_start - self.train_size
            yield train_start, test_start, test_start, test_start + self.test_size
            test_start += self.test_size

    def run(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable[..., Strategy],
        param_grid: list[dict],
        selection_metric: str = "sharpe",
    ) -> WalkForwardResult:
        fold_selections: list[dict] = []
        fold_metrics: list[dict] = []
        fold_results: list[BacktestResult] = []

        for train_start, train_end, test_start, test_end in self._folds(len(df)):
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            best_params, best_score = None, float("-inf")
            for params in param_grid:
                candidate = self.backtester.run(train_df, strategy_factory(**params))
                score = self.metrics.compute(candidate)[selection_metric]
                if pd.isna(score):
                    score = float("-inf")
                if score > best_score:
                    best_params, best_score = params, score

            test_result = self.backtester.run(test_df, strategy_factory(**best_params))

            fold_selections.append(best_params)
            fold_metrics.append(self.metrics.compute(test_result))
            fold_results.append(test_result)

        self._print_report(fold_selections, fold_metrics)

        return WalkForwardResult(
            fold_selections=fold_selections,
            fold_metrics=fold_metrics,
            stitched_result=self._stitch(fold_results),
        )

    def _stitch(self, fold_results: list[BacktestResult]) -> BacktestResult:
        """concatenate every fold's out-of-sample test segment into one continuous equity curve"""
        df = pd.concat([r.df for r in fold_results])
        positions = pd.concat([r.positions for r in fold_results])
        strategy_returns = pd.concat([r.strategy_returns for r in fold_results])
        gross_returns = pd.concat([r.gross_returns for r in fold_results])
        fee_drag = pd.concat([r.fee_drag for r in fold_results])
        slippage_drag = pd.concat([r.slippage_drag for r in fold_results])

        equity_curve = self.backtester.initial_capital * (1 + strategy_returns).cumprod()
        equity_prior = equity_curve.shift(1).fillna(self.backtester.initial_capital)
        prior_close = df["close"].shift(1)
        units = (positions * equity_prior / prior_close).fillna(0)
        cash = equity_prior * (1 - positions - fee_drag - slippage_drag)

        return BacktestResult(
            df=df,
            positions=positions,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            trade_count=sum(r.trade_count for r in fold_results),
            strategy_name=f"WalkForward({fold_results[0].strategy_name})",
            gross_returns=gross_returns,
            fee_drag=fee_drag,
            slippage_drag=slippage_drag,
            cash=cash,
            units=units,
        )

    def _print_report(self, fold_selections: list[dict], fold_metrics: list[dict]) -> None:
        print(f"{'fold':>4} | {'params':>40} | {'sharpe':>8} | {'total_return':>13}")
        print("-" * 75)
        for i, (params, metrics) in enumerate(zip(fold_selections, fold_metrics)):
            print(
                f"{i:>4} | {str(params):>40} | {metrics['sharpe']:>8.4f} | "
                f"{metrics['total_return']:>13.4f}"
            )
