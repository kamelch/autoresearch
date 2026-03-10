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

Default model is `AutoresearchLSTMRegressor` (LSTM-first). No extra package install is required for this custom model beyond your normal Freqtrade/FreqAI environment.

Optional Chronos model:

```bash
/path/to/your/freqtrade/.venv/bin/pip install chronos-forecasting
```

## 3) Backtest with FreqAI model

```bash
freqtrade backtesting \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel AutoresearchLSTMRegressor \
  --freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels \
  --timerange 20250101-20260301
```

## 4) Dry-run first (paper trading)

```bash
freqtrade trade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel AutoresearchLSTMRegressor \
  --freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels
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

## Trading Autoresearch Loop (Campaign-Isolated v2)

Use the helper runner to execute a single train/holdout candidate row with hard gates and campaign IDs:

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/run_freqtrade_backtest.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path user_data/strategies \
  --freqaimodel AutoresearchLSTMRegressor \
  --freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels \
  --timerange 20250101-20250401 \
  --campaign-id my_campaign \
  --candidate-id cand_0001 \
  --stage train \
  --description "candidate train stage"
```

Runner v2 behavior:
- campaign-aware TSV rows (`campaign_id`, `candidate_id`, `stage`, `timerange`, config/pair fingerprints)
- absolute gates before relative scoring: `profit >= min-profit-pct`, `drawdown <= max-drawdown-pct`, `sharpe >= min-sharpe`
- pair coverage gate for any pairlist length:
  - threshold = `max(pair_min_trades_floor, ceil(total_trades / (2 * num_pairs)))`
  - all non-`TOTAL` pairs must satisfy threshold
- holdout keep/discard compares only against prior `keep` rows from the same campaign and `stage=holdout`

Autonomous mode instructions are in `program.trading.md`.

### Fully automated parameter-search loop (train + holdout)

`freqai_autoresearch_loop.py` now requires explicit train/holdout ranges and evaluates each mutation in 2 stages:
1) train stage (`--train-timerange`)
2) holdout stage (`--holdout-timerange`) only if train passed
3) mutation is kept only if holdout is `keep`

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/freqai_autoresearch_loop.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path /Users/maryamsediqi/autoresearch/freqtrade/strategies \
  --freqaimodel AutoresearchLSTMRegressor \
  --freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels \
  --train-timerange 20250101-20250401 \
  --holdout-timerange 20250401-20250601 \
  --campaign-id my_campaign \
  --iterations 30
```

Optional Chronos selection for any command:

```bash
--freqaimodel AmazonChronos2Regressor \
--freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels
```

Loop v2 defaults are campaign-scoped:
- state: `freqtrade/runs/space_state_<strategy>_<campaign_id>.json`
- review log: `freqtrade/runs/review_<campaign_id>_<timestamp>.jsonl`
- adapted profile: `freqtrade/autoresearch_profile.<campaign_id>.json`

Safety behavior:
- single-run lock file for full loop duration (`lock_<strategy>_<campaign_id>.lock`)
- interrupt-safe restore (SIGINT/SIGTERM restores last committed strategy text)
- state metadata validation (`campaign_id`, `profile_hash`, `profile_path`) with auto-reset on mismatch unless `--reuse-space-state`

### One-command full automation (steps 1-6)

This command automates:
1) config install/update
2) data download
3) baseline train + holdout validation
4) iterative train/holdout optimization loop
5) final campaign holdout ranking

```bash
python3 /Users/maryamsediqi/autoresearch/scripts/automate_freqai_pipeline.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --train-timerange 20250101-20250401 \
  --holdout-timerange 20250401-20250601 \
  --iterations 30
```

Campaign defaults produced by pipeline:
- results TSV: `freqtrade/results/results_<campaign_id>.tsv`
- space state: `freqtrade/runs/space_state_<strategy>_<campaign_id>.json`
- adapted profile: `freqtrade/autoresearch_profile.<campaign_id>.json`

Optional:
- `--skip-download` if data is already present
- `--overwrite-config` to refresh config from template
- `--freqtrade-bin /path/to/your/freqtrade/.venv/bin/freqtrade` if `freqtrade` is not in PATH
- `--freqaimodel-path /Users/maryamsediqi/autoresearch/freqtrade/freqaimodels` for custom model lookup
- `--campaign-id <id>` to force a specific campaign ID
- `--profile /path/to/profile.json` to force a specific profile
- `--no-space-adaptation` to disable automatic search-space updates

### Migration note (`results.tsv` vs v2 campaign TSV)

- Legacy mixed files like `freqtrade/results.tsv` are still readable for manual inspection.
- v2 decisions are intended to run on campaign-specific TSV files (`freqtrade/results/results_<campaign_id>.tsv`) so older campaigns cannot contaminate keep/discard decisions.
