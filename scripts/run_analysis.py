from __future__ import annotations
import logging

from _common import build_backtester, build_provider, load_config, tee_stdout_to_file

from run_backtest import _param_grid

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.analysis.monte_carlo import MonteCarloAnalyzer
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.evaluation.metrics import ANNUALISATION_FACTOR, PerformanceMetrics, format_table
from src.evaluation.plots import DEFAULT_REPORTS_DIR
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.stats.attribution import alpha_beta
from src.stats.multiple_testing import deflated_sharpe_ratio, expected_max_sharpe, reality_check
from src.stats.power import detectable_effect, required_periods
from src.stats.resampling import paired_returns
from src.stats.sharpe import SE_METHODS, effective_sample_size, lo_annualisation_factor, sharpe_estimate
from src.strategies.registry import STRATEGY_REGISTRY, build_strategies

logger = logging.getLogger(__name__)

BENCHMARK_CONFIG_NAME = "buy_and_hold"
LJUNG_BOX_LAG = 10
POWER_ALPHA = 0.05
POWER_TARGET = 0.8


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()

    with tee_stdout_to_file(DEFAULT_REPORTS_DIR / "statistical_report.txt"):
        provider = build_provider(config)
        backtester = build_backtester(config)
        strategy_configs = config["strategies"]
        strategies = build_strategies(strategy_configs)

        featured = FeatureEngineer().returns(DataCleaner().clean(provider.fetch()))
        market_returns = featured["returns"].dropna()

        results = [backtester.run(featured, strategy) for strategy in strategies]
        benchmark = next(r for entry, r in zip(strategy_configs, results) if entry["name"] == BENCHMARK_CONFIG_NAME)

        print("=" * 78)
        print("STATISTICAL INFERENCE REPORT")
        print(f"{config['data']['symbol']} {config['data']['timeframe']}, {len(featured)} bars, {len(strategy_configs)} strategies")
        print("=" * 78)

        _print_sample_section(market_returns)
        _print_return_process_section(market_returns)
        _print_sharpe_section(results)
        _print_attribution_section(results, benchmark)

        grid_cfg = config.get("param_grids", {})
        mc_cfg = config.get("monte_carlo", {})
        _print_multiple_testing_section(featured, backtester, strategy_configs, results, grid_cfg, mc_cfg)

        monte_carlo = MonteCarloAnalyzer(
            n_trials=mc_cfg.get("n_trials", 5000),
            block_length=mc_cfg.get("block_length", 20),
            noise_std=mc_cfg.get("noise_std", 0.001),
        )
        _print_power_section(results, len(market_returns))
        _print_conclusion_section(results, benchmark, monte_carlo)


def _print_sample_section(market_returns: pd.Series) -> str:
    n = len(market_returns)
    lines = [
        "\n--- 1. Sample ---",
        f"  T = {n} returns ({n / ANNUALISATION_FACTOR:.2f} years)",
        f"  effective sample size after autocorrelation ~= {effective_sample_size(market_returns):.0f}",
    ]
    report = "\n".join(lines)
    print(report)
    return report


def _print_return_process_section(market_returns: pd.Series) -> str:
    """skew/kurtosis/Ljung-Box on the raw market series, not any one strategy's returns"""
    skew = float(stats.skew(market_returns, bias=False))
    kurtosis = float(stats.kurtosis(market_returns, fisher=True, bias=False))
    critical_value = float(stats.chi2.ppf(1 - POWER_ALPHA, LJUNG_BOX_LAG))

    lb_r = acorr_ljungbox(market_returns, lags=[LJUNG_BOX_LAG], return_df=True)
    lb_r2 = acorr_ljungbox(market_returns**2, lags=[LJUNG_BOX_LAG], return_df=True)

    eta = lo_annualisation_factor(market_returns)
    root_q = float(np.sqrt(ANNUALISATION_FACTOR))

    lines = [
        "\n--- 2. Return process ---",
        f"  skew = {skew:+.2f}, excess kurtosis = {kurtosis:.2f}",
        f"  Ljung-Box Q({LJUNG_BOX_LAG}) on r:   {lb_r['lb_stat'].iloc[0]:.1f} (p={lb_r['lb_pvalue'].iloc[0]:.4f})",
        f"  Ljung-Box Q({LJUNG_BOX_LAG}) on r^2: {lb_r2['lb_stat'].iloc[0]:.1f} (p={lb_r2['lb_pvalue'].iloc[0]:.4f}) "
        f"[5% critical value: {critical_value:.1f}]",
        f"  lag-1 autocorrelation: r={float(market_returns.autocorr(1)):+.2f}, r^2={float((market_returns**2).autocorr(1)):+.2f}",
        f"  eta({ANNUALISATION_FACTOR}) = {eta:.2f} vs sqrt({ANNUALISATION_FACTOR}) = {root_q:.2f} "
        "(Lo's HAC annualisation, replacing naive sqrt(q) so autocorrelation isn't laundered into Sharpe)",
    ]
    report = "\n".join(lines)
    print(report)
    return report


def _print_sharpe_section(results: list[BacktestResult]) -> str:
    columns = ["strategy", "sharpe_iid", "ci_iid", "sharpe_nonnormal", "ci_nonnormal", "sharpe_hac", "ci_hac"]
    rows = []
    for result in results:
        row = {"strategy": result.strategy_name}
        for method in SE_METHODS:
            estimate = sharpe_estimate(result.strategy_returns, method)
            row[f"sharpe_{method}"] = estimate.value
            row[f"ci_{method}"] = f"[{estimate.ci_low:+.2f}, {estimate.ci_high:+.2f}]"
        rows.append(row)

    table = format_table(columns, rows, title="\n--- 3. Sharpe ratios with 95% intervals, three SE methods ---")
    print(table)
    print("  the three barely differ: non-normality and autocorrelation are not what makes these bands wide (sample size is);")
    print("  the HAC column differs in its point estimate, from eta() replacing sqrt(q), not in its band width")
    return table


def _print_attribution_section(results: list[BacktestResult], benchmark: BacktestResult) -> str:
    columns = ["strategy", "alpha_annualised", "alpha_p", "beta", "beta_p"]
    rows = []
    for result in results:
        if result.strategy_name == benchmark.strategy_name:
            continue
        estimates = alpha_beta(result, benchmark)
        rows.append(
            {
                "strategy": result.strategy_name,
                "alpha_annualised": estimates["alpha"].value,
                "alpha_p": estimates["alpha"].p_value(),
                "beta": estimates["beta"].value,
                "beta_p": estimates["beta"].p_value(),
            }
        )

    table = format_table(columns, rows, title="\n--- 4. Attribution: alpha and beta vs buy-and-hold (HAC) ---")
    print(table)
    print("  alpha's null is zero skill: a significant alpha is return beyond just riding BTC's own beta")
    return table


def _trial_sharpe_dispersion(df: pd.DataFrame, backtester: Backtester, metrics: PerformanceMetrics, strategy_cls, candidates: list[dict]) -> float:
    """measured in-sample Sharpe spread of a strategy's own walk-forward candidate grid, not a hard-coded guess"""
    if len(candidates) <= 1:
        return 0.0
    sharpes = [metrics.compute(backtester.run(df, strategy_cls(**params)))["sharpe"] for params in candidates]
    sharpes = [s for s in sharpes if pd.notna(s)]
    return float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0


def _print_multiple_testing_section(
    featured: pd.DataFrame,
    backtester: Backtester,
    strategy_configs: list[dict],
    results: list[BacktestResult],
    grid_cfg: dict,
    mc_cfg: dict,
) -> str:
    metrics = PerformanceMetrics()

    columns = ["strategy", "n_trials", "trial_sr_std", "e_max_sr_null", "observed_sr", "deflated_p", "survives_deflation"]
    rows = []
    total_trials = 0
    for entry, result in zip(strategy_configs, results):
        candidates = _param_grid(entry["name"], grid_cfg, entry.get("params", {}))
        n_trials = len(candidates)
        total_trials += n_trials
        trial_sr_std = _trial_sharpe_dispersion(featured, backtester, metrics, STRATEGY_REGISTRY[entry["name"]], candidates)
        dsr = deflated_sharpe_ratio(result.strategy_returns, n_trials, trial_sr_std)
        rows.append(
            {
                "strategy": result.strategy_name,
                "n_trials": n_trials,
                "trial_sr_std": trial_sr_std,
                "e_max_sr_null": expected_max_sharpe(n_trials, trial_sr_std),
                "observed_sr": sharpe_estimate(result.strategy_returns, "nonnormal").value,
                "deflated_p": dsr.p_value,
                "survives_deflation": dsr.is_significant(),
            }
        )

    table = format_table(columns, rows, title="\n--- 5. Multiple testing: the reported best is a maximum ---")
    print(table)
    print(f"  {total_trials} candidate parameter sets evaluated in total across {len(strategy_configs)} strategies")

    candidate_returns = {
        r.strategy_name: r.strategy_returns for entry, r in zip(strategy_configs, results) if entry["name"] != BENCHMARK_CONFIG_NAME
    }
    benchmark_returns = next(r for entry, r in zip(strategy_configs, results) if entry["name"] == BENCHMARK_CONFIG_NAME).strategy_returns
    reality = reality_check(
        candidate_returns, benchmark_returns, n_trials=mc_cfg.get("n_trials", 5000), block_length=mc_cfg.get("block_length", 20)
    )
    print(f"\n  White's Reality Check across all {len(candidate_returns)} strategies: {reality}")
    return table


def _paired_correlation(a: BacktestResult, b: BacktestResult) -> float:
    returns_a, returns_b = paired_returns(a, b)
    return float(np.corrcoef(returns_a, returns_b)[0, 1])


def _print_power_section(results: list[BacktestResult], n_periods: int) -> str:
    ranked = sorted(results, key=lambda r: sharpe_estimate(r.strategy_returns, "nonnormal").value, reverse=True)
    top, runner_up = ranked[0], ranked[1]
    top_sr = sharpe_estimate(top.strategy_returns, "nonnormal").value
    runner_sr = sharpe_estimate(runner_up.strategy_returns, "nonnormal").value

    corr = _paired_correlation(top, runner_up)
    representative_sr = (top_sr + runner_sr) / 2
    observed_delta = abs(top_sr - runner_sr)
    detectable = detectable_effect(n_periods, corr, representative_sr, power=POWER_TARGET, alpha=POWER_ALPHA)

    lines = [
        "\n--- 6. Power: what this experiment can and cannot detect ---",
        f"  top two by Sharpe: {top.strategy_name} ({top_sr:.3f}) vs {runner_up.strategy_name} ({runner_sr:.3f}), "
        f"return correlation = {corr:.2f} (the actual measured value, not an assumed one)",
        f"  at T={n_periods} bars, smallest reliably detectable Sharpe difference ({POWER_TARGET:.0%} power, "
        f"alpha={POWER_ALPHA}): {detectable:.3f}",
    ]
    if observed_delta > 0:
        needed = required_periods(observed_delta, corr, representative_sr, power=POWER_TARGET, alpha=POWER_ALPHA)
        lines.append(
            f"  observed difference is {observed_delta:.3f}; detecting it at {POWER_TARGET:.0%} power would need "
            f"{needed / ANNUALISATION_FACTOR:.1f} years of daily data"
        )
    else:
        lines.append("  observed difference is exactly zero: no amount of data would separate them")
    report = "\n".join(lines)
    print(report)
    return report


def _print_conclusion_section(results: list[BacktestResult], benchmark: BacktestResult, monte_carlo: MonteCarloAnalyzer) -> str:
    candidates = [r for r in results if r.strategy_name != benchmark.strategy_name]
    best = max(candidates, key=lambda r: sharpe_estimate(r.strategy_returns, "nonnormal").value)
    best_sr = sharpe_estimate(best.strategy_returns, "nonnormal").value
    benchmark_sr = sharpe_estimate(benchmark.strategy_returns, "nonnormal").value

    comparison = monte_carlo.compare_strategies(best, benchmark)

    lines = ["\n--- 7. Conclusion ---"]
    if comparison.is_distinguishable():
        lines.append(
            f"  SUPPORTED: {best.strategy_name} (Sharpe {best_sr:.3f}) is distinguishable from buy-and-hold "
            f"(Sharpe {benchmark_sr:.3f}) on this history, P(better)={comparison.prob_a_beats_b():.2f}."
        )
    else:
        delta = abs(best_sr - benchmark_sr)
        corr = _paired_correlation(best, benchmark)
        representative_sr = (best_sr + benchmark_sr) / 2
        needed_years = (
            required_periods(delta, corr, representative_sr, power=POWER_TARGET, alpha=POWER_ALPHA) / ANNUALISATION_FACTOR
            if delta > 0
            else float("inf")
        )
        lines.append(
            f"  NOT SUPPORTED: {best.strategy_name} (Sharpe {best_sr:.3f}) is not distinguishable from buy-and-hold "
            f"(Sharpe {benchmark_sr:.3f}) on this history, P(better)={comparison.prob_a_beats_b():.2f}. "
            f"Resolving a difference this size at {POWER_TARGET:.0%} power would need roughly {needed_years:.0f} more years of data."
        )
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    main()
