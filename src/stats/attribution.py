from __future__ import annotations
import statsmodels.api as sm

from src.backtest.result import BacktestResult
from src.evaluation.metrics import ANNUALISATION_FACTOR
from src.stats.estimate import Estimate
from src.stats.resampling import paired_returns
from src.stats.sharpe import newey_west_lags


def alpha_beta(strategy_result: BacktestResult, benchmark_result: BacktestResult, max_lags: int | None = None) -> dict[str, Estimate]:
    """
    OLS of strategy returns on benchmark returns with HAC standard errors
    """
    strategy_returns, benchmark_returns = paired_returns(strategy_result, benchmark_result)
    lags = newey_west_lags(len(strategy_returns)) if max_lags is None else max_lags

    design = sm.add_constant(benchmark_returns)
    fit = sm.OLS(strategy_returns, design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    alpha, beta = fit.params
    alpha_se, beta_se = fit.bse

    return {
        # alpha is a per-period mean excess return, so it annualises by the factor itself, not its square root
        "alpha": Estimate.from_se(value=float(alpha) * ANNUALISATION_FACTOR, se=float(alpha_se) * ANNUALISATION_FACTOR, method="HAC"),
        "beta": Estimate.from_se(value=float(beta), se=float(beta_se), method="HAC"),
    }
