# Crypto Trading Framework

A systematic trading research framework for cryptocurrency markets.
The system fetches exchange market data, generates signals from a
library of interchangeable strategies, and evaluates them through a
fee-aware, vectorised backtesting engine with out-of-sample validation,
benchmarked against buy-and-hold.

## Strategies

All strategies implement a common interface and are configured, run,
and compared through the same pipeline. The library spans opposing
strategy families, since each is a bet on different market behaviour:

**Trend-following**
- *MA crossover* - enters long when a fast moving average crosses above
  a slow one (recent prices overtaking the longer-term average) and
  exits to cash on the reverse cross. Because each average dilutes
  single-day noise by a factor of its window length, only sustained
  moves trigger a crossover, filtering volatility at the cost of
  delayed entries and exits.
- *Time-series momentum* - long while price exceeds its level from a
  fixed lookback period ago.

**Mean reversion**
- *RSI mean reversion* - enters long when RSI signals oversold
  conditions and exits as it normalises.

**Breakout**
- *Donchian channel breakout* - enters long when price escapes above
  its recent trading range, exits when it breaks below.

**Signal filters and ensembles**
- *RSI filter* - a composable wrapper that suppresses any strategy's
  entries into overbought conditions.
- *Voting ensembles* - combine multiple strategies by hard voting
  (long when at least k members agree) or soft voting (position sized
  by the weighted average of member signals). Ensemble value depends on
  members with uncorrelated errors, so the framework reports the
  pairwise return correlations of ensemble members.

## Pipeline

1. **Data** - historical OHLCV candles (open, high, low, close, volume)
   for any exchange-listed pair (e.g. BTC/USDT) via the ccxt library.
2. **Signals** - each strategy maps price data to target positions
   (long, flat, or fractional).
3. **Backtest** - a vectorised engine simulates positions against
   historical returns, applying per-trade fees and enforcing a one-bar
   execution delay to eliminate lookahead bias.
4. **Validation** - walk-forward analysis selects parameters on
   training windows and evaluates on unseen data, so reported
   performance is out-of-sample; parameter sweeps favour robust
   plateaus over fragile single-point optima.
5. **Evaluation** - performance metrics (total and annualised return,
   volatility, Sharpe ratio, maximum drawdown, trade count) reported
   for every strategy alongside the buy-and-hold benchmark, with equity
   curve and signal charts.