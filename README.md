# Crypto Trading Algorithm

A systematic trading framework for cryptocurrency markets. The system
fetches exchange market data, generates signals from a trend-following
strategy, and evaluates performance through backtesting against buy-and-hold strategy.

## Strategy

The core strategy is a moving average (MA) crossover, a momentum-based
trend-following approach:

- Two moving averages are computed over the closing price: a fast MA
  (short window) tracking recent price action, and a slow MA (long
  window) tracking the broader trend.
- When the fast MA crosses above the slow MA, recent prices have
  overtaken the longer-term average, so the system enters a long position.
- When the fast MA crosses back below, momentum has faded, then the system exits to cash.

Because each average dilutes single-day noise by a factor of its window
length, only sustained price moves can trigger a crossover. The signal
therefore filters out short-term volatility at the cost of delayed
entries and exits (the standard trade-off of trend-following systems).

## Pipeline

1. **Data** - historical OHLCV candles (open, high, low, close, volume)
   for any exchange-listed pair (e.g. BTC/USDT) via the ccxt library.
2. **Signals** - the strategy maps price data to target positions (long or flat).
3. **Backtest** - a vectorised engine simulates the positions against
   historical returns and enforcing a one-bar execution delay to eliminate lookahead bias.
4. **Evaluation** - performance metrics (total and annualised return,
   volatility, Sharpe ratio, maximum drawdown, trade count) reported
   alongside the buy-and-hold benchmark, with equity curve and signal charts.