from __future__ import annotations

from _common import build_backtester, build_provider, load_config

from src.evaluation.metrics import PerformanceMetrics
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.ma_crossover import MACrossoverStrategy


def main() -> None:
    config = load_config()

    raw = build_provider(config).fetch()
    df = FeatureEngineer().returns(DataCleaner().clean(raw))

    backtester = build_backtester(config)
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
