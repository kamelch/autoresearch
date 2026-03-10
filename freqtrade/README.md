# Freqtrade Integration (Futures + Long/Short)

This folder connects this repo to Freqtrade using FreqAI.
Default tuning assumptions:
- Exchange: Binance futures
- Pairs: BTC/USDT:USDT, ETH/USDT:USDT
- Timeframe: 5m
- Risk profile: conservative (2 max trades, 1.0x leverage in strategy)

## 1) Put files into your Freqtrade user dir

From your Freqtrade project directory:

```bash
mkdir -p user_data/strategies
cp /Users/maryamsediqi/autoresearch/freqtrade/strategies/AutoresearchFreqAIStrategy.py user_data/strategies/
cp /Users/maryamsediqi/autoresearch/freqtrade/config/config_freqai_autoresearch.example.json user_data/config_freqai_autoresearch.json
```

Edit `user_data/config_freqai_autoresearch.json`:
- exchange keys
- whitelist pairs
- stake settings
- keep `trading_mode: futures` and `margin_mode: isolated` unless you intentionally change mode

## 2) Download market data

```bash
freqtrade download-data \
  --config user_data/config_freqai_autoresearch.json \
  --timeframes 5m 15m 1h \
  --timerange 20240101-20260301
```

## 3) Backtest with FreqAI model

```bash
freqtrade backtesting \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-20260301
```

## 4) Dry-run first (paper trading)

```bash
freqtrade trade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel PyTorchMLPRegressor
```

## 5) Live trading switch

Only after stable dry-run results:
- set `"dry_run": false`
- set exchange API keys
- start with smallest stake and strict risk limits

## Notes

- This starter supports long and short (`can_short = true`).
- Entry/exit uses both predicted return and probability gates (`pred_prob_up`).
- Key tune knobs in strategy:
  - `long_entry_ret`, `short_entry_ret`
  - `long_prob_min`, `short_prob_max`
  - `long_exit_prob_max`, `short_exit_prob_min`
- Retraining is configured with `live_retrain_hours` in config.
- For real deployment, tune `minimal_roi`, `stoploss`, and feature/label settings from backtest evidence.

## Trading Autoresearch Loop (Karpathy-style)

Use the helper runner in this repo to execute one experiment and auto-log metrics:

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/run_freqtrade_backtest.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-20260301 \
  --description "test change"
```

This script:
- runs `freqtrade backtesting`
- exports JSON to `freqtrade/runs/`
- parses key metrics
- computes `score = profit_pct - 0.5 * max_drawdown_pct`
- appends `freqtrade/results.tsv`
- prints `suggestion: keep|discard|crash`

Autonomous mode instructions are in `program.trading.md`.

### Fully automated parameter-search loop

To continuously mutate strategy parameters and keep only winners:

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/freqai_autoresearch_loop.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path /Users/maryamsediqi/autoresearch/freqtrade/strategies \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-20260301 \
  --iterations 30 \
  --baseline-if-empty
```

By default this loop mutates parameters defined in:
- `freqtrade/autoresearch_profile.example.json`

And for each iteration it:
- changes one parameter
- runs `scripts/run_freqtrade_backtest.py`
- reads `suggestion` from `freqtrade/results.tsv`
- keeps the file change only when suggestion is `keep`
- reverts file change for `discard`/`crash`

### One-command full automation (steps 1-6)

This command automates:
1) config install/update
2) data download
3) baseline run
4) iterative optimization loop
5) final top-results summary

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/automate_freqai_pipeline.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --timerange 20250101-20260301 \
  --iterations 30
```

Optional:
- `--skip-download` if data is already present
- `--overwrite-config` to refresh config from template
