# Crypto Trading Framework

A systematic trading research framework for cryptocurrency markets.
The system fetches exchange market data, generates signals from a
library of interchangeable strategies, and evaluates them through a
fee- and slippage-aware, vectorised backtesting engine with
out-of-sample validation, benchmarked against buy-and-hold.

## Strategies

All strategies implement a common interface and are configured, run,
and compared through the same pipeline. The library spans opposing
strategy families, since each is a bet on different market behaviour:

**Trend-following**
- MA crossover - enters long when a fast moving average crosses above
  a slow one (recent prices overtaking the longer-term average) and
  exits to cash on the reverse cross. Because each average dilutes
  single-day noise by a factor of its window length, only sustained
  moves trigger a crossover, filtering volatility at the cost of
  delayed entries and exits.
- Time-series momentum - long while price exceeds its level from a
  fixed lookback period ago.

**Mean reversion**
- RSI mean reversion - enters long when RSI signals oversold
  conditions and exits as it normalises.

**Breakout**
- Donchian channel breakout - enters long when price escapes above
  its recent trading range, exits when it breaks below.

**Signal filters and ensembles**
- RSI filter - a composable wrapper that suppresses any strategy's
  entries into overbought conditions.
- Voting ensembles - combine multiple strategies by hard voting
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
   historical returns on a real cash/holdings ledger, applying
   per-trade fees and slippage (an adverse price-impact cost, in basis
   points, proportional to the size of each position change) and
   enforcing a one-bar execution delay to eliminate lookahead bias.
   Reports itemise gross return, fee drag, and slippage drag
   separately so it's clear exactly where performance is lost. An
   optional minimum holding period can suppress single-bar whipsaw
   flips.
4. **Validation** - walk-forward analysis selects parameters on
   training windows and evaluates on unseen data, so reported
   performance is out-of-sample; parameter sweeps favour robust
   plateaus over fragile single-point optima.
5. **Evaluation** - performance metrics (total and annualised return,
   volatility, Sharpe ratio, maximum drawdown, trade count) reported
   for every strategy alongside the buy-and-hold benchmark, with equity
   curve and signal charts.
6. **Monte Carlo analysis** - Monte Carlo analysis, enabled via
   `config.yaml`'s `monte_carlo.enabled` flag: block bootstrap
   resampling of a strategy's daily returns produces DISTRIBUTIONS of
   final return and max drawdown (not just the single historical
   numbers as a single backtest is one draw from many the same daily
   behaviour could have produced, and drawdown is especially
   order-dependent), plus a noise-perturbation check that re-runs the
   strategy on many small random perturbations of the price series to
   reveal how much a strategy's edge depends on the exact historical
   path.


## Backtesting
Run with:
```
python scripts/run_backtest.py
```

## Paper Trading

Trading is simulated bookkeeping
written to a log file, useful for watching a strategy's behaviour on
live data before (if ever) trusting it further.

Run it with:
```
python scripts/paper_trade.py
```

It polls the exchange on a schedule matching `config.yaml`'s
`data.timeframe`, and recomputes the strategy configured under `paper_trading.strategy_name` 
over the full history seen so far, using the exact same `Backtester` as the research backtests
above. Every decision is appended as a JSON line to
`paper_trading.log_path` (default `logs/paper_trades.jsonl`), and a
status line is printed on every poll. Network failures retry with
exponential backoff and never crash the loop.

After a paper trading session has run for a while, reconcile the
logged equity path against a retrospective historical backtest of the
same period:

```
python scripts/reconcile_paper_trade.py logs/paper_trades.jsonl
```

Divergences between the two paths reveal what the backtest abstraction
hides (partial bars, exchange data revisions, and poll-timing gaps)
not a bug in the backtest engine itself.

### Results

BTC/USDT daily candles, full available history - 3,281 bars
(2017-08-17 to 2026-08-10), 0.1% fee, 5 bps slippage, 10,000 starting
capital.

**In-sample (plain backtest).** Returns are net of fees and slippage.
"Round trips" counts completed entry-to-exit trades; P/L is realised
over those trades.

| Strategy | Total return | Sharpe | Max drawdown | Round trips | Win rate | Profit factor | P/L |
|---|---|---|---|---|---|---|---|
| TSMOM (60) | +2,995% | **1.05** | -60.6% | 78 | 32% | 2.02 | +$300,656 |
| Donchian breakout (20,10) | +2,033% | 1.04 | -55.7% | 65 | 45% | 1.88 | +$203,305 |
| Soft voting ensemble | +846% | 1.03 | -53.1% | 58 | 38% | **2.33** | +$86,009 |
| MA crossover (20,50) | +1,133% | 0.83 | -75.0% | 36 | 39% | 1.75 | +$115,912 |
| Hard voting (k=2) | +98% | 0.41 | **-44.1%** | 117 | 58% | 1.19 | +$9,821 |
| Buy and hold (benchmark) | +1,391% | 0.79 | -83.2% | 0 | - | - | - (still open) |

**Out-of-sample (walk-forward).** Parameters selected on each training
window, evaluated on unseen data. The fold geometry is itself a
parameter, so both train windows are reported rather than picking the
flattering one:

| Strategy | Sharpe (365-bar train, 32 folds) | Sharpe (730-bar train, 28 folds) |
|---|---|---|
| TSMOM | **1.19** | 1.13 |
| Soft voting ensemble | 0.88 | **1.16** |
| MA crossover | 0.75 | 0.79 |
| Donchian breakout | 0.75 | 0.68 |
| Buy and hold (benchmark) | 0.73 | 0.74 |
| RSI-filtered crossover | 0.28 | 0.56 |
| RSI mean reversion | 0.27 | 0.58 |

