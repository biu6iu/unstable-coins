from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.analysis.monte_carlo import MonteCarloAnalyzer
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider
from src.evaluation.metrics import PerformanceMetrics
from src.evaluation.plots import ReportPlotter
from src.pipeline import Pipeline
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.ensemble import VotingStrategy, print_correlation_matrix
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.registry import STRATEGY_REGISTRY, build_strategies
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.volatility_breakout import DonchianBreakoutStrategy
from src.validation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")

def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_provider(config: dict):
    data_cfg = config["data"]
    ccxt_provider = CCXTDataProvider(
        symbol=data_cfg["symbol"],
        timeframe=data_cfg["timeframe"],
        limit=data_cfg["limit"],
    )
    try:
        ccxt_provider.fetch() 
        return ccxt_provider
    except Exception as exc:
        logger.warning("CCXT fetch failed (%s)", exc)
        return SyntheticDataProvider(n_periods=data_cfg["limit"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = _load_config()

    provider = _build_provider(config)

    backtest_cfg = config["backtest"]
    backtester = Backtester(
        fee=backtest_cfg["fee"],
        initial_capital=backtest_cfg["initial_capital"],
        slippage_bps=backtest_cfg.get("slippage_bps", 5.0),
        min_holding_period=backtest_cfg.get("min_holding_period", 0),
    )

    strategy_configs = config["strategies"]
    strategies = build_strategies(strategy_configs)

    # example ensemble: trend-following + mean reversion + breakout, so
    # members win/lose in different regimes rather than restating the same
    # bet three times (see the correlation diagnostic below). Inserted
    # before buy_and_hold so buy_and_hold stays the last/benchmark entry
    # the pipeline plots against.
    ensemble = VotingStrategy(
        [
            MACrossoverStrategy(fast=20, slow=50),
            RSIMeanReversionStrategy(window=14, buy_below=30, exit_above=50),
            DonchianBreakoutStrategy(entry_window=20, exit_window=10),
        ],
        k=2,
    )
    bh_index = next(i for i, s in enumerate(strategies) if s.name == "BuyAndHold")
    strategies.insert(bh_index, ensemble)

    metrics = PerformanceMetrics()

    mc_cfg = config.get("monte_carlo", {})
    monte_carlo = (
        MonteCarloAnalyzer(
            n_trials=mc_cfg.get("n_trials", 5000),
            block_length=mc_cfg.get("block_length", 20),
            noise_std=mc_cfg.get("noise_std", 0.001),
        )
        if mc_cfg.get("enabled", False)
        else None
    )

    pipeline = Pipeline(
        provider=provider,
        cleaner=DataCleaner(),
        engineer=FeatureEngineer(),
        strategies=strategies,
        backtester=backtester,
        metrics=metrics,
        plotter=ReportPlotter(),
        monte_carlo=monte_carlo,
    )

    plain_res = pipeline.run(plot_filename="strategy_comparison.png")
    _print_cost_impact_table(plain_res)
    _report_ensemble_diagnostics(ensemble, plain_res, monte_carlo)

    # provider.fetch() is a cache hit so this reconstructs the exact df pipeline used internally without a second network round-trip
    df = FeatureEngineer().returns(DataCleaner().clean(provider.fetch()))
    wf_res = _run_walk_forward(df, backtester, metrics, strategy_configs, config["walk_forward"])

    _print_combined_table(plain_res, wf_res, metrics)


def _report_ensemble_diagnostics(ensemble: VotingStrategy, plain_res: list[BacktestResult], monte_carlo: MonteCarloAnalyzer | None) -> None:
    """
    1. hard voting can flip more often than any single member (fee bleed)
    2. the diversification claim rests on members errors being uncorrelated

    both need checking explicitly
    """
    res_by_name = {result.strategy_name: result for result in plain_res}
    ensemble_result = res_by_name[ensemble.name]
    member_res = [res_by_name[member.name] for member in ensemble.strategies]

    member_trade_counts = ", ".join(f"{r.strategy_name}={r.trade_count}" for r in member_res)
    print(
        f"\nEnsemble trade count: {ensemble.name} = {ensemble_result.trade_count} trades "
        f"(members: {member_trade_counts}; max single member = "
        f"{max(r.trade_count for r in member_res)})"
    )

    print_correlation_matrix(member_res + [ensemble_result])

    if monte_carlo is not None:
        comparisons = monte_carlo.compare_ensemble_drawdowns(ensemble_result, member_res)
        monte_carlo.print_drawdown_comparison(comparisons)


def _run_walk_forward(df, backtester: Backtester, metrics: PerformanceMetrics, strategy_configs: list[dict], wf_cfg: dict) -> list[BacktestResult]:
    """Run every registered strategy through WalkForwardValidator"""
    validator = WalkForwardValidator(
        backtester=backtester,
        metrics=metrics,
        train_size=wf_cfg["train_size"],
        test_size=wf_cfg["test_size"],
        expanding=wf_cfg.get("expanding", False),
    )

    stitched_res = []
    for entry in strategy_configs:
        strategy_cls = STRATEGY_REGISTRY[entry["name"]]
        params = entry.get("params", {})
        print(f"\nWalk-forward folds: {entry['name']}")
        wf_result = validator.run(
            df,
            strategy_factory=strategy_cls,
            param_grid=[params],
            selection_metric="sharpe",
        )
        stitched_res.append(wf_result.stitched_result)
    return stitched_res


def _print_cost_impact_table(res: list[BacktestResult]) -> None:
    columns = ["strategy", "gross_return", "fee_drag", "slippage_drag", "net_return"]
    rows = []
    for result in res:
        gross_return = (1 + result.gross_returns).prod() - 1
        net_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
        rows.append(
            {
                "strategy": result.strategy_name,
                "gross_return": gross_return,
                "fee_drag": result.fee_drag.sum(),
                "slippage_drag": result.slippage_drag.sum(),
                "net_return": net_return,
            }
        )

    widths = {c: max(len(c), *(len(f"{row[c]:.4f}" if c != "strategy" else row[c]) for row in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print("\nCost impact (before/after fees + slippage):")
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        cells = [
            row["strategy"].ljust(widths["strategy"]),
            *(f"{row[c]:.4f}".ljust(widths[c]) for c in columns[1:]),
        ]
        print(" | ".join(cells))


def _print_combined_table(plain_res: list[BacktestResult], wf_res: list[BacktestResult], metrics: PerformanceMetrics) -> None:
    columns = ["mode", "strategy", "total_return", "sharpe", "max_drawdown", "trade_count"]
    rows = []
    for mode, res in (("plain", plain_res), ("walk_forward", wf_res)):
        for result in res:
            row = metrics.compute(result)
            row["mode"] = mode
            rows.append(row)

    def _fmt(value) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    widths = {c: max(len(c), *(len(_fmt(row[c])) for row in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print("\nCombined comparison (plain backtest vs walk-forward out-of-sample):")
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(_fmt(row[c]).ljust(widths[c]) for c in columns))


if __name__ == "__main__":
    main()
