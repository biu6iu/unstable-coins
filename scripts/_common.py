from __future__ import annotations
import logging
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.backtest.engine import Backtester
from src.data.base import DataProvider
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")


class _Tee:
    """Writes to multiple streams at once, so stdout can be mirrored to a file"""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout_to_file(path: Path | str):
    """Mirror everything printed to stdout into `path` as well as the terminal, so a run's full report output is saved alongside being shown live"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f, redirect_stdout(_Tee(sys.stdout, f)):
        yield


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
