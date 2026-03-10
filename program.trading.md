# trading-autoresearch

Autonomous research loop for Freqtrade + FreqAI.

## Scope

You are optimizing this strategy stack:
- `freqtrade/strategies/AutoresearchFreqAIStrategy.py`
- `freqtrade/config/config_freqai_autoresearch.example.json`

You may also modify:
- `scripts/build_ml_dataset_from_ohlcv.py`

Do not modify:
- `prepare.py`
- `train.py`

## Goal

Maximize trading quality on backtests using this score:

`score = profit_pct - 0.5 * max_drawdown_pct`

Higher is better.

Secondary preferences:
- More stable Sharpe / Sortino
- Fewer overfit symptoms across timeranges
- Simpler logic when performance is similar

## Setup

1. Ensure Freqtrade is installed and data is downloaded.
2. Create a fresh branch for this run, e.g. `autoresearch/trading-mar10`.
3. Initialize `freqtrade/results.tsv` with header (already in repo).

## Baseline run

Run one baseline experiment before any edits:

```bash
python3 scripts/run_freqtrade_backtest.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path /Users/maryamsediqi/autoresearch/freqtrade/strategies \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-20260301 \
  --description baseline
```

## Experiment loop

Repeat forever:

1. Pick one idea and edit only a small surface area.
2. Commit the change.
3. Run backtest via `scripts/run_freqtrade_backtest.py`.
4. Check runner output (`status`, `score`, `suggestion`).
5. If suggestion is `keep`: keep commit and continue from it.
6. If suggestion is `discard`: reset to previous keep commit.
7. If suggestion is `crash`: inspect log, fix if trivial, otherwise discard.

Alternative: run the automation helper:

```bash
python3 scripts/freqai_autoresearch_loop.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --config user_data/config_freqai_autoresearch.json \
  --strategy AutoresearchFreqAIStrategy \
  --strategy-path /Users/maryamsediqi/autoresearch/freqtrade/strategies \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-20260301 \
  --iterations 30 \
  --baseline-if-empty
```

The helper mutates one strategy parameter per iteration and auto-reverts non-kept changes.

Single-command end-to-end automation is also available:

```bash
python3 scripts/automate_freqai_pipeline.py \
  --freqtrade-dir /path/to/your/freqtrade \
  --timerange 20250101-20260301 \
  --iterations 30
```

If `freqtrade` is not in PATH, add:
`--freqtrade-bin /path/to/your/freqtrade/.venv/bin/freqtrade`

## Decision policy

- Keep only changes that improve score by a meaningful margin.
- Prefer robust wins over tiny wins likely due to noise.
- Periodically validate with alternate timeranges:
  - `20240101-20250101`
  - `20250101-20260301`

## Notes

- If backtest command fails, status is recorded as `crash`.
- Results are appended to `freqtrade/results.tsv` automatically.
- Run logs and exported JSON are saved under `freqtrade/runs/`.
