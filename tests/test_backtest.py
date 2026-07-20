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

    result = Backtester(fee=0.0, initial_capital=100.0, slippage_bps=0.0).run(df, strategy)

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

    result = Backtester(fee=fee, initial_capital=initial_capital, slippage_bps=0.0).run(
        df, strategy
    )

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

    result = Backtester(fee=fee, initial_capital=initial_capital, slippage_bps=0.0).run(
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


def test_gross_returns_and_fee_drag_reconcile_with_equity_curve():
    # Hand-computed: position = [0, 1, 0], asset_returns = [0, 0.2, -0.25].
    close = [100, 120, 90]
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([1, 0, 1])
    fee = 0.01
    initial_capital = 1000.0

    result = Backtester(fee=fee, initial_capital=initial_capital, slippage_bps=0.0).run(
        df, strategy
    )

    assert list(result.gross_returns) == pytest.approx([0, 0.2, 0])
    assert list(result.fee_drag) == pytest.approx([0, 0.01, 0.01])
    assert list(result.slippage_drag) == pytest.approx([0, 0, 0])

    expected_equity = initial_capital * (
        1 + result.gross_returns - result.fee_drag - result.slippage_drag
    ).cumprod()
    pd.testing.assert_series_equal(
        result.equity_curve, expected_equity, check_names=False
    )
    assert result.equity_curve.iloc[-1] == pytest.approx(1178.1)


def test_cash_and_units_ledger_balances_to_equity_curve():
    # Same scenario as above, hand-computed: buy 10 units at bar 1 (cost
    # 1000 from equity_prior=1000, price=100), pay a $10 fee from cash,
    # then mark to market and unwind back to cash at bar 2.
    close = [100, 120, 90]
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([1, 0, 1])
    fee = 0.01
    initial_capital = 1000.0

    result = Backtester(fee=fee, initial_capital=initial_capital, slippage_bps=0.0).run(
        df, strategy
    )

    assert list(result.units) == pytest.approx([0, 10, 0])
    assert list(result.cash) == pytest.approx([1000, -10, 1178.1])

    ledger_equity = result.cash + result.units * df["close"]
    pd.testing.assert_series_equal(
        ledger_equity, result.equity_curve, check_names=False
    )


def test_slippage_charged_exactly_on_each_position_change():
    # Two trades (enter at bar 1, exit at bar 2), hand-computed: each trade
    # is charged fee + slippage on the notional traded, same mechanism as
    # the existing fee test but with a non-zero slippage_bps.
    close = [100, 100, 100, 100]
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([1, 0, 0, 0])
    fee = 0.001
    slippage_bps = 10.0  # 10 bps = 0.001
    initial_capital = 1000.0

    result = Backtester(
        fee=fee, initial_capital=initial_capital, slippage_bps=slippage_bps
    ).run(df, strategy)

    assert list(result.positions) == [0, 1, 0, 0]
    assert list(result.slippage_drag) == pytest.approx([0, 0.001, 0.001, 0])

    per_trade_drag = fee + slippage_bps / 10_000
    assert result.equity_curve.iloc[1] == pytest.approx(
        initial_capital * (1 - per_trade_drag)
    )
    assert result.equity_curve.iloc[2] == pytest.approx(
        initial_capital * (1 - per_trade_drag) ** 2
    )
    assert result.equity_curve.iloc[3] == pytest.approx(
        initial_capital * (1 - per_trade_drag) ** 2
    )


def test_total_return_reconciles_with_gross_minus_fees_minus_slippage():
    # Same scenario as test_gross_returns_and_fee_drag_reconcile_with_equity_curve,
    # now with slippage also in effect: position = [0, 1, 0].
    close = [100, 120, 90]
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([1, 0, 1])
    fee = 0.01
    slippage_bps = 100.0  # 100 bps = 0.01, same magnitude as fee here
    initial_capital = 1000.0

    result = Backtester(
        fee=fee, initial_capital=initial_capital, slippage_bps=slippage_bps
    ).run(df, strategy)

    assert list(result.gross_returns) == pytest.approx([0, 0.2, 0])
    assert list(result.fee_drag) == pytest.approx([0, 0.01, 0.01])
    assert list(result.slippage_drag) == pytest.approx([0, 0.01, 0.01])

    expected_equity = initial_capital * (
        1 + result.gross_returns - result.fee_drag - result.slippage_drag
    ).cumprod()
    pd.testing.assert_series_equal(
        result.equity_curve, expected_equity, check_names=False
    )
    assert result.equity_curve.iloc[-1] == pytest.approx(1156.4)


def test_min_holding_period_suppresses_single_bar_whipsaw_flips():
    # signal = [0,1,0,1,0,1,0] -> raw shifted position = [0,0,1,0,1,0,1],
    # flipping every bar from index 2 onward. With min_holding_period=3,
    # the first two flips (indices 2 and 3) are within 3 bars of the
    # implicit bar-0 "change" and are suppressed; the flip at index 4 is
    # the first one 3+ bars out, so it's accepted and held through the end
    # since every later flip is again too soon after it.
    close = [100] * 7
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([0, 1, 0, 1, 0, 1, 0])

    result = Backtester(
        fee=0.0, initial_capital=100.0, slippage_bps=0.0, min_holding_period=3
    ).run(df, strategy)

    assert list(result.positions) == [0, 0, 0, 0, 1, 1, 1]
    assert result.trade_count == 1


def test_min_holding_period_zero_leaves_positions_unaffected():
    close = [100] * 7
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([0, 1, 0, 1, 0, 1, 0])

    result = Backtester(
        fee=0.0, initial_capital=100.0, slippage_bps=0.0, min_holding_period=0
    ).run(df, strategy)

    assert list(result.positions) == [0, 0, 1, 0, 1, 0, 1]
