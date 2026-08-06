from __future__ import annotations
import argparse
import logging

import _common  # noqa: F401 - side effect only: puts the repo root on sys.path

import pandas as pd

from src.backtest.engine import Backtester
from src.data.ccxt_provider import CCXTDataProvider
from src.live.paper_trader import diff_equity_paths, load_decision_log, parse_timeframe_seconds
from src.strategies.registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_BARS = 200


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to the paper-trading JSONL log")
    parser.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS,
        help="Extra history to fetch before the logged period, so lookback-window "
        "strategies (e.g. moving averages) aren't handicapped by a truncated warm-up",
    )
    args = parser.parse_args()

    header, decisions = load_decision_log(args.log_path)
    if decisions.empty:
        print("No decisions logged yet, nothing to reconcile.")
        return

    strategy = _build_strategy(header)
    hist_df = _fetch_reconciliation_window(header, decisions, args.warmup_bars)

    backtester = Backtester(
        fee=header["fee"], initial_capital=header["initial_capital"], slippage_bps=header["slippage_bps"]
    )
    result = backtester.run(hist_df, strategy)

    diff = diff_equity_paths(decisions["equity"], result.equity_curve)
    _print_summary(diff)


def _build_strategy(header: dict):
    registry_name = header.get("strategy_registry_name")
    if not registry_name:
        raise ValueError(
            "Log has no strategy_registry_name in its session header "
            "(logged by an older PaperTrader run without strategy_config) - cannot rebuild the strategy"
        )
    return STRATEGY_REGISTRY[registry_name](**header.get("strategy_params", {}))


def _fetch_reconciliation_window(header: dict, decisions: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    """fetch the logged period plus a warm-up buffer, so early logged bars aren't handicapped by truncated lookback"""
    timeframe_seconds = parse_timeframe_seconds(header["timeframe"])
    since = decisions.index[0] - pd.Timedelta(seconds=timeframe_seconds * warmup_bars)
    total_bars = warmup_bars + len(decisions) + 5

    provider = CCXTDataProvider(
        symbol=header["symbol"],
        timeframe=header["timeframe"],
        exchange_id=header["exchange_id"],
        limit=total_bars,
        since=since,
        use_cache=False,
    )
    return provider.fetch()


def _print_summary(diff) -> None:
    print(f"\nReconciliation over {len(diff)} overlapping bars:")
    print(f"  mean abs diff: {diff['abs_diff'].abs().mean():,.4f}")
    print(f"  max abs diff:  {diff['abs_diff'].abs().max():,.4f}")
    print(f"  mean pct diff: {diff['pct_diff'].abs().mean():.4%}")
    print(f"  max pct diff:  {diff['pct_diff'].abs().max():.4%}")

    worst = diff.reindex(diff["pct_diff"].abs().sort_values(ascending=False).index).head(5)
    print("\n  worst 5 divergent bars:")
    print(worst[["logged_equity", "backtested_equity", "abs_diff", "pct_diff"]].to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\n  Divergences here typically trace back to partial bars, exchange data "
        "revisions, or poll-timing gaps - not a bug in the backtest engine itself."
    )

if __name__ == "__main__":
    main()