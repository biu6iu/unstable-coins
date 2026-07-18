import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy


def _make_ohlcv(close):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2022-01-01", periods=len(close), freq="D", name="timestamp")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 100.0),
        },
        index=index,
    )


class _FixedSignalStrategy(Strategy):
    """Test double: emits a caller-supplied signal instead of computing one."""

    def __init__(self, signal):
        self._signal = signal

    @property
    def name(self) -> str:
        return "Fixed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = self._signal
        return out


def test_signal_at_bar_t_earns_return_of_bar_t_plus_1_not_bar_t():
    close = [100, 110, 90, 130]
    df = _make_ohlcv(close)
    # long only at bar index 1
    strategy = _FixedSignalStrategy([0, 1, 0, 0])

    result = Backtester(fee=0.0, initial_capital=100.0).run(df, strategy)

    # position is signal shifted by one bar: position[2] = signal[1] = 1.
    assert list(result.positions) == [0, 0, 1, 0]

    # check no lookahead 
    expected_bar2_return = close[2] / close[1] - 1
    assert result.strategy_returns.iloc[2] == pytest.approx(expected_bar2_return)
    assert result.strategy_returns.iloc[0] == 0
    assert result.strategy_returns.iloc[1] == 0
    assert result.strategy_returns.iloc[3] == 0


def test_fee_charged_exactly_on_each_position_change():
    close = [100, 100, 100, 100]  
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([1, 0, 0, 0])
    fee = 0.001
    initial_capital = 1000.0

    result = Backtester(fee=fee, initial_capital=initial_capital).run(df, strategy)

    assert list(result.positions) == [0, 1, 0, 0]
    assert result.trade_count == 2

    assert result.equity_curve.iloc[1] == pytest.approx(initial_capital * (1 - fee))
    assert result.equity_curve.iloc[2] == pytest.approx(initial_capital * (1 - fee) ** 2)
    assert result.equity_curve.iloc[3] == pytest.approx(initial_capital * (1 - fee) ** 2)


def test_buy_and_hold_equity_matches_asset_return_minus_one_entry_fee():
    close = [100, 130]
    df = _make_ohlcv(close)
    fee = 0.001
    initial_capital = 10000.0

    result = Backtester(fee=fee, initial_capital=initial_capital).run(
        df, BuyAndHoldStrategy()
    )

    asset_total_return = close[-1] / close[0] - 1
    expected_final_equity = initial_capital * (1 + asset_total_return - fee)

    assert result.trade_count == 1
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_equity)


def test_result_carries_strategy_name_and_input_df():
    close = [100, 110, 120]
    df = _make_ohlcv(close)

    result = Backtester().run(df, BuyAndHoldStrategy())

    assert result.strategy_name == "BuyAndHold"
    pd.testing.assert_frame_equal(result.df, df)
