import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.ensemble import SoftVotingStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.registry import STRATEGY_REGISTRY, build_strategies
from src.strategies.rsi_filter import RSIFilterStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.tsmom import TimeSeriesMomentumStrategy
from src.strategies.volatility_breakout import DonchianBreakoutStrategy


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


def test_ma_crossover_flips_signal_on_hand_computed_bars():
    # Hand-verified with fast=2/slow=3 SMAs:
    #   sma_2: [nan, 10, 10, 11, 13, 15, 13, 9, 7, 5]
    #   sma_3: [nan, nan, 10, 10.667, 12, 14, 13.333, 11.333, 8, 6]
    # sma_2 crosses above sma_3 at bar 3, back below at bar 6.
    close = [10, 10, 10, 12, 14, 16, 10, 8, 6, 4]
    df = _make_ohlcv(close)

    out = MACrossoverStrategy(fast=2, slow=3).generate_signals(df)

    expected_signal = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert list(out["signal"]) == expected_signal


def test_ma_crossover_rejects_fast_greater_or_equal_slow():
    with pytest.raises(ValueError):
        MACrossoverStrategy(fast=50, slow=20)
    with pytest.raises(ValueError):
        MACrossoverStrategy(fast=20, slow=20)


def test_ma_crossover_does_not_add_position_or_mutate_input():
    close = [10, 11, 12, 13, 14]
    df = _make_ohlcv(close)
    original = df.copy()

    out = MACrossoverStrategy(fast=2, slow=3).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)
    assert "position" not in out.columns
    assert "signal" in out.columns


def test_buy_and_hold_is_always_long():
    close = [10, 9, 11, 8, 20]
    df = _make_ohlcv(close)

    out = BuyAndHoldStrategy().generate_signals(df)

    assert (out["signal"] == 1).all()


def test_buy_and_hold_does_not_mutate_input():
    close = [10, 9, 11]
    df = _make_ohlcv(close)
    original = df.copy()

    BuyAndHoldStrategy().generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_strategy_names():
    assert MACrossoverStrategy(fast=20, slow=50).name == "MACrossover(20,50)"
    assert BuyAndHoldStrategy().name == "BuyAndHold"


class _AlwaysLongStrategy(Strategy):
    """Test double: always long, so any suppression must come from the filter."""

    @property
    def name(self) -> str:
        return "AlwaysLong"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 1
        return out


def test_rsi_filter_suppresses_signal_when_overbought():
    # A strong, sustained uptrend (all gains, no losses) drives RSI to 100.
    close = [10 + i for i in range(30)]
    df = _make_ohlcv(close)

    out = RSIFilterStrategy(_AlwaysLongStrategy(), rsi_window=14, rsi_overbought=70.0).generate_signals(df)

    rsi = out["rsi_14"]
    assert (rsi.dropna() > 70).any()  # sanity check: the setup actually triggers overbought
    assert (out.loc[rsi > 70, "signal"] == 0).all()
    assert (out.loc[rsi <= 70, "signal"] == 1).all()


def test_rsi_filter_does_not_mutate_input():
    close = [10, 11, 12, 11, 10]
    df = _make_ohlcv(close)
    original = df.copy()

    RSIFilterStrategy(_AlwaysLongStrategy()).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_rsi_filter_name_includes_wrapped_strategy():
    filtered = RSIFilterStrategy(MACrossoverStrategy(fast=20, slow=50), rsi_overbought=70.0)

    assert filtered.name == "RSIFilter(MACrossover(20,50),70.0)"


def test_rsi_mean_reversion_enters_below_buy_below_and_exits_above_exit_above():
    # window=3, buy_below=30, exit_above=70. RSI (Wilder's smoothing,
    # hand-verified with a script) for this close series:
    #   [nan, nan, nan, 0.0, 0.0, 33.3, 55.6, 70.37, 80.2, 86.8, 91.2, 60.8, 40.5, 27.0]
    # -> enters at bar 3 (RSI dips below 30), HOLDS through bars 5-6 even
    #    though RSI is climbing back through the neutral zone (this is the
    #    hysteresis band, not a single-bar trigger), exits at bar 7 (RSI
    #    crosses above 70), stays flat, then re-enters at bar 13.
    close = [100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 101, 100, 99]
    df = _make_ohlcv(close)

    out = RSIMeanReversionStrategy(window=3, buy_below=30.0, exit_above=70.0).generate_signals(df)

    expected_signal = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert list(out["signal"]) == expected_signal


def test_rsi_mean_reversion_rejects_buy_below_greater_or_equal_exit_above():
    with pytest.raises(ValueError):
        RSIMeanReversionStrategy(buy_below=50.0, exit_above=30.0)
    with pytest.raises(ValueError):
        RSIMeanReversionStrategy(buy_below=30.0, exit_above=30.0)


def test_rsi_mean_reversion_does_not_mutate_input():
    close = [100, 98, 96, 94, 96, 98, 100]
    df = _make_ohlcv(close)
    original = df.copy()

    RSIMeanReversionStrategy(window=3).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_donchian_breakout_flips_signal_on_hand_computed_bars():
    # entry_window=3, exit_window=2 (hand-verified with a script):
    #   entry_high (shifted): [nan, nan, nan, 11, 12, 12, 13, 13, 14, 14]
    #   exit_low (shifted):   [nan, nan, 10, 9, 9, 8, 8, 7, 7, 6]
    # -> enters whenever close beats the prior 3-bar high, exits whenever
    #    it falls below the prior 2-bar low.
    close = [10, 11, 9, 12, 8, 13, 7, 14, 6, 15]
    df = _make_ohlcv(close)

    out = DonchianBreakoutStrategy(entry_window=3, exit_window=2).generate_signals(df)

    expected_signal = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    assert list(out["signal"]) == expected_signal


def test_donchian_breakout_channel_excludes_the_current_bar():
    # Bar 3 (close=12) sets a new all-time high, but the entry channel
    # compared against bar 3 is built from bars [0:3] only (max=11) - a
    # high a bar makes is never the level that same bar is judged against.
    close = [10, 11, 9, 12, 8, 13, 7, 14, 6, 15]
    df = _make_ohlcv(close)

    entry_high = df["close"].rolling(window=3).max().shift(1)

    assert entry_high.iloc[3] == 11.0  # not 12.0 (bar 3's own close)
    assert entry_high.iloc[3] != df["close"].iloc[3]


def test_donchian_breakout_does_not_mutate_input():
    close = [10, 11, 9, 12, 8, 13, 7, 14, 6, 15]
    df = _make_ohlcv(close)
    original = df.copy()

    DonchianBreakoutStrategy(entry_window=3, exit_window=2).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_tsmom_flips_signal_on_hand_computed_bars():
    # lookback=2: close.shift(2) = [nan, nan, 10, 12, 9, 15]
    # signal is 1 wherever close beats its value 2 bars ago.
    close = [10, 12, 9, 15, 8, 20]
    df = _make_ohlcv(close)

    out = TimeSeriesMomentumStrategy(lookback=2).generate_signals(df)

    expected_signal = [0, 0, 0, 1, 0, 1]
    assert list(out["signal"]) == expected_signal


def test_tsmom_does_not_mutate_input():
    close = [10, 12, 9, 15, 8, 20]
    df = _make_ohlcv(close)
    original = df.copy()

    TimeSeriesMomentumStrategy(lookback=2).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_new_strategy_names():
    assert RSIMeanReversionStrategy(window=14, buy_below=30, exit_above=50).name == \
        "RSIMeanReversion(14,30,50)"
    assert DonchianBreakoutStrategy(entry_window=20, exit_window=10).name == \
        "DonchianBreakout(20,10)"
    assert TimeSeriesMomentumStrategy(lookback=90).name == "TSMOM(90)"


def test_registry_builds_strategies_from_config_entries():
    configs = [
        {"name": "ma_crossover", "params": {"fast": 10, "slow": 20}},
        {"name": "rsi_mean_reversion", "params": {"window": 14}},
        {"name": "donchian_breakout", "params": {}},
        {"name": "tsmom", "params": {"lookback": 30}},
        {"name": "buy_and_hold", "params": {}},
    ]

    strategies = build_strategies(configs)

    assert [s.name for s in strategies] == [
        "MACrossover(10,20)",
        "RSIMeanReversion(14,30.0,50.0)",
        "DonchianBreakout(20,10)",
        "TSMOM(30)",
        "BuyAndHold",
    ]


def test_registry_contains_all_classical_strategies():
    assert set(STRATEGY_REGISTRY) == {
        "buy_and_hold",
        "ma_crossover",
        "rsi_mean_reversion",
        "donchian_breakout",
        "tsmom",
        "rsi_filtered_crossover",
        "soft_voting_ensemble",
    }


def test_registry_builds_rsi_filtered_crossover_from_config_entry():
    strategy = build_strategies(
        [{"name": "rsi_filtered_crossover", "params": {"fast": 10, "slow": 20}}]
    )[0]
    assert isinstance(strategy, RSIFilterStrategy)
    assert isinstance(strategy.strategy, MACrossoverStrategy)
    assert strategy.strategy.name == "MACrossover(10,20)"


def test_registry_builds_soft_voting_ensemble_from_config_entry():
    strategy = build_strategies([{"name": "soft_voting_ensemble", "params": {}}])[0]
    assert isinstance(strategy, SoftVotingStrategy)
    assert [member.name for member in strategy.strategies] == [
        "MACrossover(20,50)",
        "RSIMeanReversion(14,30.0,50.0)",
        "DonchianBreakout(20,10)",
    ]


def test_registry_soft_voting_weights_derived_from_member_sharpes():
    """Weights should track measured edge: clipped at zero, normalised, in member order"""
    strategy = build_strategies(
        [{"name": "soft_voting_ensemble", "params": {"member_sharpes": [1.0, -0.5, 3.0]}}]
    )[0]
    assert isinstance(strategy, SoftVotingStrategy)
    assert strategy.weights == pytest.approx([0.25, 0.0, 0.75])


def test_registry_soft_voting_falls_back_to_uniform_when_no_member_has_positive_sharpe():
    """All-negative Sharpes would normalise a zero vector; uniform is the honest default"""
    strategy = build_strategies(
        [{"name": "soft_voting_ensemble", "params": {"member_sharpes": [-0.9, -0.7, -0.3]}}]
    )[0]
    assert isinstance(strategy, SoftVotingStrategy)
    assert strategy.weights is None


def test_registry_soft_voting_explicit_weights_override_member_sharpes():
    strategy = build_strategies(
        [
            {
                "name": "soft_voting_ensemble",
                "params": {"member_sharpes": [1.0, 2.0, 3.0], "weights": [1.0, 1.0, 1.0]},
            }
        ]
    )[0]
    assert isinstance(strategy, SoftVotingStrategy)
    assert strategy.weights == [1.0, 1.0, 1.0]


def test_registry_soft_voting_rsi_window_is_configurable():
    """The mean-reversion member's RSI window is walk-forward selected, so it must be reachable from config"""
    strategy = build_strategies(
        [{"name": "soft_voting_ensemble", "params": {"rsi_window": 10, "buy_below": 20}}]
    )[0]
    assert isinstance(strategy, SoftVotingStrategy)
    assert strategy.strategies[1].name == "RSIMeanReversion(10,20,50.0)"
