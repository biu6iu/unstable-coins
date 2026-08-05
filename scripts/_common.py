"""Shared CLI setup for scripts/run_backtest.py and scripts/param_sweep.py"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.backtest.engine import Backtester
from src.data.base import DataProvider
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_provider(config: dict) -> DataProvider:
    """CCXT provider with a synthetic-data fallback so scripts still run offline"""
    data_cfg = config["data"]
    provider = CCXTDataProvider(
        symbol=data_cfg["symbol"], timeframe=data_cfg["timeframe"], limit=data_cfg["limit"]
    )
    try:
        provider.fetch()
        return provider
    except Exception as exc:
        logger.warning("CCXT fetch failed (%s)", exc)
        return SyntheticDataProvider(n_periods=data_cfg["limit"])


def build_backtester(config: dict) -> Backtester:
    backtest_cfg = config["backtest"]
    return Backtester(
        fee=backtest_cfg["fee"],
        initial_capital=backtest_cfg["initial_capital"],
        slippage_bps=backtest_cfg.get("slippage_bps", 5.0),
        min_holding_period=backtest_cfg.get("min_holding_period", 0),
    )
