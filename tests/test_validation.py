import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.evaluation.metrics import PerformanceMetrics
from src.strategies.base import Strategy
from src.validation.walk_forward import WalkForwardValidator


def _make_ohlcv(close, index=None):
    close = np.asarray(close, dtype=float)
    if index is None:
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


class _PrescribedSignalStrategy(Strategy):

    def __init__(self, signal: pd.Series):
        self._signal = signal

    @property
    def name(self) -> str:
        return "Prescribed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = self._signal.reindex(df.index)
        return out


def _validator(train_size=5, test_size=5, expanding=False, fee=0.0):
    return WalkForwardValidator(
        backtester=Backtester(fee=fee, initial_capital=1000.0),
        metrics=PerformanceMetrics(),
        train_size=train_size,
        test_size=test_size,
        expanding=expanding,
    )


def test_no_test_window_ever_overlaps_its_train_window():
    for train_start, train_end, test_start, test_end in _validator(
        train_size=5, test_size=3
    )._folds(n_bars=20):
        assert train_end == test_start
        assert train_start < train_end
        assert test_start < test_end


def test_no_test_window_ever_overlaps_its_train_window_expanding():
    for train_start, train_end, test_start, test_end in _validator(
        train_size=5, test_size=3, expanding=True
    )._folds(n_bars=20):
        assert train_start == 0
        assert train_end == test_start
        assert test_start < test_end


def test_stitched_equity_curve_length_equals_sum_of_test_windows():
    # n_bars=20, train_size=5, test_size=5 -> folds at [5:10],[10:15],[15:20]
    # -> 3 folds * 5 bars = 15 stitched bars.
    close = 100 + np.cumsum(np.full(20, 1.0))
    df = _make_ohlcv(close)
    signal = pd.Series(1, index=df.index)

    result = _validator(train_size=5, test_size=5).run(
        df,
        strategy_factory=lambda: _PrescribedSignalStrategy(signal),
        param_grid=[{}],
        selection_metric="total_return",
    )

    assert len(result.fold_selections) == 3
    assert len(result.stitched_result.equity_curve) == 15


def test_selection_uses_only_train_data_by_construction():
    # 10 bars, train=[0:5], test=[5:10], price rising steadily throughout.
    # signal_a is long during train / flat during test; signal_b is the
    # opposite. A wins on train (positive return vs B's 0%), so the
    # validator must select A - even though B would clearly have won if
    # it had been allowed to peek at the test window (B earns a positive
    # return there while the selected A earns exactly 0%).
    close = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
    df = _make_ohlcv(close)

    signal_a = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=df.index)
    signal_b = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], index=df.index)

    result = _validator(train_size=5, test_size=5).run(
        df,
        strategy_factory=lambda signal: _PrescribedSignalStrategy(signal),
        param_grid=[{"signal": signal_a}, {"signal": signal_b}],
        selection_metric="total_return",
    )

    assert len(result.fold_selections) == 1
    assert result.fold_selections[0]["signal"] is signal_a
    # Out-of-sample: the selected (train-optimal) strategy is flat
    # throughout the test window, so it earns exactly 0% there - the
    # train-window selection carries through untouched by what would
    # have scored best on test.
    assert result.fold_metrics[0]["total_return"] == pytest.approx(0.0)
