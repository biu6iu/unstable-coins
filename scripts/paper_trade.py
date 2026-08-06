from __future__ import annotations
import logging

from _common import build_backtester, load_config

from src.data.ccxt_provider import CCXTDataProvider
from src.live.paper_trader import PaperTrader
from src.strategies.registry import build_strategies

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    data_cfg = config["data"]
    pt_cfg = config["paper_trading"]

    # use_cache=False: a live loop must see the exchange's current state on every poll
    provider = CCXTDataProvider(
        symbol=data_cfg["symbol"],
        timeframe=data_cfg["timeframe"],
        limit=data_cfg["limit"],
        use_cache=False,
    )
    backtester = build_backtester(config)
    strategy_config = _find_strategy_config(config["strategies"], pt_cfg["strategy_name"])
    strategy = build_strategies([strategy_config])[0]

    trader = PaperTrader(
        provider=provider,
        strategy=strategy,
        backtester=backtester,
        log_path=pt_cfg["log_path"],
        strategy_config=strategy_config,
        poll_interval_seconds=pt_cfg.get("poll_interval_seconds"),
        max_retries=pt_cfg.get("max_retries", 5),
        retry_base_delay_seconds=pt_cfg.get("retry_base_delay_seconds", 2.0),
        retry_max_delay_seconds=pt_cfg.get("retry_max_delay_seconds", 60.0),
    )
    trader.run_forever()


def _find_strategy_config(strategy_configs: list[dict], strategy_name: str) -> dict:
    """reuse the strategy's params from config's strategies, so they're only defined once"""
    matches = [entry for entry in strategy_configs if entry["name"] == strategy_name]
    if not matches:
        raise ValueError(f"paper_trading.strategy_name {strategy_name!r} not found in config's strategies list")
    return matches[0]


if __name__ == "__main__":
    main()
