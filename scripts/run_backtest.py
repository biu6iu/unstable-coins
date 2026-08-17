from __future__ import annotations
import itertools
import logging
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from _common import build_backtester, build_provider, load_config, tee_stdout_to_file

import pandas as pd

from src.analysis.monte_carlo import MonteCarloAnalyzer
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.data.base import DataProvider
from src.evaluation.metrics import PerformanceMetrics, format_table
from src.evaluation.plots import DEFAULT_REPORTS_DIR, ReportPlotter
from src.evaluation.trades import extract_trades, trade_summary
from src.pipeline import Pipeline
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy
from src.strategies.ensemble import VotingStrategy, print_correlation_matrix
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.registry import STRATEGY_REGISTRY, build_strategies
from src.strategies.rsi_filter import RSIFilterStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.volatility_breakout import DonchianBreakoutStrategy
from src.validation.walk_forward import WalkForwardValidator, fold_count

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()

    # prints to reports/appendix_backtest.txt
    with tee_stdout_to_file(DEFAULT_REPORTS_DIR / "appendix_backtest.txt"):
        provider = build_provider(config)
        backtester = build_backtester(config)
        strategy_configs = config["strategies"]
        strategies, ensemble = _build_strategies(strategy_configs)

        metrics = PerformanceMetrics()
        monte_carlo = _build_monte_carlo(config)
        pipeline = _build_pipeline(provider, backtester, strategies, metrics, monte_carlo)

        plain_res = pipeline.run(plot_filename="strategy_comparison.png")
        _print_cost_impact_table(plain_res)
        _print_trade_summary_table(plain_res)
        _save_trade_logs(plain_res, DEFAULT_REPORTS_DIR)
        _report_ensemble_diagnostics(ensemble, plain_res, monte_carlo)
        _report_best_strategy_robustness(plain_res, metrics, monte_carlo)
        _report_best_vs_field(plain_res, metrics, monte_carlo)

        wf_cfg = config["walk_forward"]
        # provider.fetch() is a cache hit so this reconstructs the exact df pipeline used internally without a second network round-trip
        wf_df = FeatureEngineer().returns(DataCleaner().clean(provider.fetch()))
        wf_by_train_size = {
            train_size: _run_walk_forward(
                wf_df,
                backtester,
                metrics,
                strategy_configs,
                wf_cfg,
                config.get("param_grids", {}),
                train_size,
            )
            for train_size in _train_sizes(wf_cfg)
        }

        _print_geometry_comparison(wf_by_train_size, metrics, len(wf_df), wf_cfg)
        _print_combined_table(plain_res, wf_by_train_size[wf_cfg["train_size"]], metrics)


def build_hard_voting_ensemble() -> VotingStrategy:
    return VotingStrategy(
        [
            RSIFilterStrategy(MACrossoverStrategy(fast=20, slow=50), rsi_window=14, rsi_overbought=70.0),
            RSIMeanReversionStrategy(window=10, buy_below=20, exit_above=50),
            DonchianBreakoutStrategy(entry_window=20, exit_window=10),
        ],
        k=2,
    )


def _build_strategies(strategy_configs: list[dict]) -> tuple[list[Strategy], VotingStrategy]:
    """
    Build the config/registry-driven strategies
    """
    strategies = build_strategies(strategy_configs)

    ensemble = build_hard_voting_ensemble()
    bh_index = next(i for i, s in enumerate(strategies) if s.name == "BuyAndHold")
    strategies.insert(bh_index, ensemble)
    return strategies, ensemble


def _build_monte_carlo(config: dict) -> MonteCarloAnalyzer | None:
    """Monte Carlo analysis is opt-in via config's monte_carlo.enabled flag"""
    mc_cfg = config.get("monte_carlo", {})
    if not mc_cfg.get("enabled", False):
        return None
    return MonteCarloAnalyzer(
        n_trials=mc_cfg.get("n_trials", 5000),
        block_length=mc_cfg.get("block_length", 20),
        noise_std=mc_cfg.get("noise_std", 0.001),
    )


def _build_pipeline(
    provider: DataProvider,
    backtester: Backtester,
    strategies: list[Strategy],
    metrics: PerformanceMetrics,
    monte_carlo: MonteCarloAnalyzer | None,
) -> Pipeline:
    return Pipeline(
        provider=provider,
        cleaner=DataCleaner(),
        engineer=FeatureEngineer(),
        strategies=strategies,
        backtester=backtester,
        metrics=metrics,
        plotter=ReportPlotter(),
        monte_carlo=monte_carlo,
    )


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


def _report_best_strategy_robustness(plain_res: list[BacktestResult], metrics: PerformanceMetrics, monte_carlo: MonteCarloAnalyzer | None) -> None:
    if monte_carlo is None:
        return

    scored = [(metrics.compute(result)["sharpe"], result) for result in plain_res]
    best_sharpe, best = max(scored, key=lambda pair: (pd.notna(pair[0]), pair[0]))

    print(f"\nRobustness check on the best plain-backtest Sharpe ({best_sharpe:.4f}):")
    monte_carlo.print_bootstrap_report(monte_carlo.bootstrap(best), label=best.strategy_name)


def _report_best_vs_field(plain_res: list[BacktestResult], metrics: PerformanceMetrics, monte_carlo: MonteCarloAnalyzer | None) -> None:
    """
    whether the headline strategy is separable from the two comparisons a reader will actually make: the benchmark, and the next best strategy
    """
    if monte_carlo is None:
        return

    scored = [(metrics.compute(result)["sharpe"], result) for result in plain_res]
    ranked = sorted((pair for pair in scored if pd.notna(pair[0])), key=lambda pair: pair[0], reverse=True)
    if len(ranked) < 2:
        return

    best, runner_up = ranked[0][1], ranked[1][1]
    print(f"\nIs the best strategy ({best.strategy_name}) distinguishable from the field?")

    benchmark = next((r for r in plain_res if r.strategy_name == "BuyAndHold"), None)
    if benchmark is not None and benchmark is not best:
        monte_carlo.print_comparison_report(monte_carlo.compare_strategies(best, benchmark))
    monte_carlo.print_comparison_report(monte_carlo.compare_strategies(best, runner_up))


_GRID_CONSTRAINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "ma_crossover": (("fast", "slow"),),
    "rsi_mean_reversion": (("buy_below", "exit_above"),),
    "donchian_breakout": (("exit_window", "entry_window"),),
}


def _is_valid_combo(name: str, params: dict) -> bool:
    """Constraints only apply when the grid actually varies both sides of a pair"""
    return all(
        params[a] < params[b]
        for a, b in _GRID_CONSTRAINTS.get(name, ())
        if a in params and b in params
    )


def _param_grid(name: str, grid_cfg: dict, params: dict) -> list[dict]:
    """
    Expand a strategy's declared option lists into every valid combination.
    """
    options = grid_cfg.get(name)
    if not options:
        return [params]

    keys = list(options)
    combos = [
        {**params, **dict(zip(keys, values))}
        for values in itertools.product(*(options[k] for k in keys))
    ]
    valid = [c for c in combos if _is_valid_combo(name, c)]
    if not valid:
        raise ValueError(f"param_grids.{name} expands to no valid combinations")
    return valid


def _train_sizes(wf_cfg: dict) -> list[int]:
    """
    train windows to evaluate, configured one first
    """
    primary = wf_cfg["train_size"]
    variants = wf_cfg.get("train_size_variants") or [primary]
    return [primary] + [size for size in variants if size != primary]


def _run_walk_forward(
    df: pd.DataFrame,
    backtester: Backtester,
    metrics: PerformanceMetrics,
    strategy_configs: list[dict],
    wf_cfg: dict,
    grid_cfg: dict,
    train_size: int,
) -> list[BacktestResult]:
    """Run every registered strategy through WalkForwardValidator"""
    validator = WalkForwardValidator(
        backtester=backtester,
        metrics=metrics,
        train_size=train_size,
        test_size=wf_cfg["test_size"],
        expanding=wf_cfg.get("expanding", False),
    )

    window = "expanding" if wf_cfg.get("expanding", False) else "rolling"
    print(
        f"\n=== Walk-forward geometry: {len(df)} bars, train={train_size}, "
        f"test={wf_cfg['test_size']}, {window} -> {validator.fold_count(len(df))} folds ==="
    )

    stitched_res = []
    for entry in strategy_configs:
        strategy_cls = STRATEGY_REGISTRY[entry["name"]]
        param_grid = _param_grid(entry["name"], grid_cfg, entry.get("params", {}))
        print(f"\nWalk-forward folds: {entry['name']} ({len(param_grid)} candidate(s) per fold)")
        wf_result = validator.run(
            df,
            strategy_factory=strategy_cls,
            param_grid=param_grid,
            selection_metric="sharpe",
        )
        _print_parameter_stability(wf_result.fold_selections)
        stitched_res.append(wf_result.stitched_result)
    return stitched_res


def _print_parameter_stability(fold_selections: list[dict]) -> None:
    """
    Summarise how often the folds agreed on a parameter set 
    """
    # keyed on the rendered dict rather than its items
    counts = Counter(str(dict(sorted(params.items()))) for params in fold_selections)
    modal, agreed = counts.most_common(1)[0]
    verdict = "stable" if agreed > len(fold_selections) / 2 else "UNSTABLE"
    print(
        f"  parameter stability: {agreed}/{len(fold_selections)} folds chose "
        f"{modal} ({len(counts)} distinct choices) -> {verdict}"
    )


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

    print(format_table(columns, rows, title="\nCost impact (before/after fees + slippage):"))


def _print_trade_summary_table(res: list[BacktestResult]) -> None:
    columns = ["strategy", "num_trades", "win_rate", "avg_win", "avg_loss", "profit_factor", "total_pnl_dollars"]
    rows = []
    for result in res:
        summary = trade_summary(extract_trades(result))
        rows.append({"strategy": result.strategy_name, **{c: summary[c] for c in columns[1:]}})

    print(format_table(columns, rows, title="\nPer-trade P/L summary:"))


def _save_trade_logs(res: list[BacktestResult], output_dir: Path) -> None:
    """Write each strategy's full closed-trade log to a CSV, for deeper offline analysis"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in res:
        trades = extract_trades(result)
        filename = re.sub(r"[^A-Za-z0-9]+", "_", result.strategy_name).strip("_") + "_trades.csv"
        pd.DataFrame([asdict(t) for t in trades]).to_csv(output_dir / filename, index=False)


def _print_geometry_comparison(
    wf_by_train_size: dict[int, list[BacktestResult]],
    metrics: PerformanceMetrics,
    n_bars: int,
    wf_cfg: dict,
) -> None:
    """
    Out-of-sample results under each fold geometry, side by side.
    """
    columns = ["train_size", "folds", "strategy", "total_return", "sharpe", "max_drawdown"]
    rows = []
    for train_size, results in wf_by_train_size.items():
        folds = fold_count(n_bars, train_size, wf_cfg["test_size"], wf_cfg.get("expanding", False))
        for result in results:
            row = metrics.compute(result)
            row["train_size"] = train_size
            row["folds"] = folds
            rows.append(row)

    print(format_table(columns, rows, title="\nWalk-forward out-of-sample by fold geometry:"))


def _print_combined_table(plain_res: list[BacktestResult], wf_res: list[BacktestResult], metrics: PerformanceMetrics) -> None:
    columns = ["mode", "strategy", "total_return", "sharpe", "max_drawdown", "trade_count"]
    rows = []
    for mode, res in (("plain", plain_res), ("walk_forward", wf_res)):
        for result in res:
            row = metrics.compute(result)
            row["mode"] = mode
            rows.append(row)

    print(format_table(columns, rows, title="\nCombined comparison (plain backtest vs walk-forward out-of-sample):"))


if __name__ == "__main__":
    main()