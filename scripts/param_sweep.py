from __future__ import annotations
import sys
from pathlib import Path
import yaml

from src.backtest.engine import Backtester
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider
from src.evaluation.metrics import PerformanceMetrics
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.ma_crossover import MACrossoverStrategy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG_PATH = Path("config/config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _fetch_data(config: dict):
    data_cfg = config["data"]
    provider = CCXTDataProvider(
        symbol=data_cfg["symbol"], timeframe=data_cfg["timeframe"], limit=data_cfg["limit"]
    )
    try:
        return provider.fetch()
    except Exception:
        return SyntheticDataProvider(n_periods=data_cfg["limit"]).fetch()


def main() -> None:
    config = _load_config()

    raw = _fetch_data(config)
    df = FeatureEngineer().returns(DataCleaner().clean(raw))

    backtest_cfg = config["backtest"]
    backtester = Backtester(
        fee=backtest_cfg["fee"],
        initial_capital=backtest_cfg["initial_capital"],
        slippage_bps=backtest_cfg.get("slippage_bps", 5.0),
        min_holding_period=backtest_cfg.get("min_holding_period", 0),
    )
    metrics = PerformanceMetrics()

    sweep_cfg = config["sweep"]
    rows = []
    for fast in sweep_cfg["fast_options"]:
        for slow in sweep_cfg["slow_options"]:
            # invalid combination
            if fast >= slow:
                continue  
            result = backtester.run(df, MACrossoverStrategy(fast=fast, slow=slow))
            row = metrics.compute(result)
            row["fast"] = fast
            row["slow"] = slow
            rows.append(row)

    header = (
        f"{'fast':>6} | {'slow':>6} | {'total_return':>13} | "
        f"{'sharpe':>8} | {'max_drawdown':>13} | {'trades':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['fast']:>6} | {row['slow']:>6} | {row['total_return']:>13.4f} | "
            f"{row['sharpe']:>8.4f} | {row['max_drawdown']:>13.4f} | {row['trade_count']:>6}"
        )


if __name__ == "__main__":
    main()
