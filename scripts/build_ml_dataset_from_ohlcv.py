#!/usr/bin/env python3
"""Build a supervised ML dataset from OHLCV candles.

Supports CSV / Parquet / Feather input (as exported by many trading tools).
Output is a CSV with engineered features and a 3-class label:
- 1: buy
- 0: hold
- -1: sell

Example:
  uv run python scripts/build_ml_dataset_from_ohlcv.py \
    --input user_data/data/binance/BTC_USDT-5m.feather \
    --output data/btcusdt_5m_ml.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Freqtrade data usually includes: date, open, high, low, close, volume
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, math.nan)


def normal_cdf_series(z: pd.Series) -> pd.Series:
    sqrt2 = math.sqrt(2.0)
    return z.map(lambda v: 0.5 * (1.0 + math.erf(v / sqrt2)) if pd.notna(v) else math.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = out["close"]
    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    close_safe = close.replace(0, math.nan)
    open_safe = open_.replace(0, math.nan)

    # Returns / momentum.
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_12"] = out["close"].pct_change(12)
    out["ret_48"] = out["close"].pct_change(48)
    out["log_ret_1"] = close_safe.map(math.log).diff(1)
    out["log_ret_3"] = close_safe.map(math.log).diff(3)
    out["mom_12_48"] = out["ret_12"] - out["ret_48"]

    # Candle geometry.
    out["hl_spread"] = (high - low) / close_safe
    out["oc_spread"] = (close - open_) / open_safe
    candle_range = (high - low).replace(0, math.nan)
    out["body_to_range"] = (close - open_).abs() / candle_range
    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick_frac"] = (high - candle_top) / candle_range
    out["lower_wick_frac"] = (candle_bottom - low) / candle_range

    # Volume regime features.
    out["vol_chg_1"] = volume.pct_change(1)
    out["vol_chg_12"] = volume.pct_change(12)
    out["vol_sma_20"] = volume.rolling(20).mean()
    out["vol_ratio_20"] = volume / out["vol_sma_20"].replace(0, math.nan)
    out["vol_z_20"] = rolling_zscore(volume, 20)
    out["dollar_volume"] = close * volume
    out["dollar_vol_z_20"] = rolling_zscore(out["dollar_volume"], 20)

    # Trend.
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["ema_ratio_12_26"] = out["ema_12"] / out["ema_26"].replace(0, math.nan) - 1.0
    out["ema_ratio_12_50"] = out["ema_12"] / out["ema_50"].replace(0, math.nan) - 1.0
    out["rsi_7"] = rsi(close, 7)
    out["rsi_14"] = rsi(close, 14)
    out["rsi_spread"] = out["rsi_7"] - out["rsi_14"]

    # Volatility / range position.
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()
    out["atr_pct_14"] = out["atr_14"] / close_safe
    out["ret_std_12"] = out["ret_1"].rolling(12).std()
    out["ret_std_48"] = out["ret_1"].rolling(48).std()
    out["ret_z_48"] = rolling_zscore(out["ret_1"], 48)
    out["close_sma_20"] = close.rolling(20).mean()
    out["close_std_20"] = close.rolling(20).std()
    out["bb_width"] = (2.0 * out["close_std_20"]) / out["close_sma_20"].replace(0, math.nan)
    bb_upper = out["close_sma_20"] + 2.0 * out["close_std_20"]
    bb_lower = out["close_sma_20"] - 2.0 * out["close_std_20"]
    out["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, math.nan)
    out["roll_high_24"] = high.rolling(24).max()
    out["roll_low_24"] = low.rolling(24).min()
    out["range_pos_24"] = (close - out["roll_low_24"]) / (
        out["roll_high_24"] - out["roll_low_24"]
    ).replace(0, math.nan)

    # Time features (cyclical encoding usually works better than raw ints).
    out["hour"] = out["date"].dt.hour
    out["weekday"] = out["date"].dt.dayofweek
    hour_angle = (2.0 * math.pi * out["hour"]) / 24.0
    weekday_angle = (2.0 * math.pi * out["weekday"]) / 7.0
    out["hour_sin"] = hour_angle.map(math.sin)
    out["hour_cos"] = hour_angle.map(math.cos)
    out["weekday_sin"] = weekday_angle.map(math.sin)
    out["weekday_cos"] = weekday_angle.map(math.cos)

    return out


def add_probability_statistics_features(
    df: pd.DataFrame,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    stats_lookback: int,
) -> pd.DataFrame:
    """Add leakage-safe statistical and probability features."""
    out = df.copy()

    # Rolling distribution stats of 1-step return.
    out["ret_mean_12"] = out["ret_1"].rolling(12).mean()
    out["ret_mean_48"] = out["ret_1"].rolling(48).mean()
    out["ret_mean_lb"] = out["ret_1"].rolling(stats_lookback).mean()
    out["ret_std_lb"] = out["ret_1"].rolling(stats_lookback).std()
    out["ret_skew_lb"] = out["ret_1"].rolling(stats_lookback).skew()
    out["ret_kurt_lb"] = out["ret_1"].rolling(stats_lookback).kurt()
    out["price_z_lb"] = rolling_zscore(out["close"], stats_lookback)
    out["volume_z_lb"] = rolling_zscore(out["volume"], stats_lookback)

    # Empirical up/down probabilities from recent window.
    up_event = (out["ret_1"] > 0).astype(float)
    down_event = (out["ret_1"] < 0).astype(float)
    out["prob_up_emp_lb"] = up_event.rolling(stats_lookback).mean()
    out["prob_down_emp_lb"] = down_event.rolling(stats_lookback).mean()
    p = out["prob_up_emp_lb"].clip(lower=1e-12, upper=1 - 1e-12)
    out["direction_entropy_lb"] = -(p * p.map(math.log) + (1.0 - p) * (1.0 - p).map(math.log))

    # Normal approximation for horizon return probabilities.
    # Uses rolling mean/std of 1-step log returns and scales by horizon.
    log_ret_1 = out["close"].replace(0, math.nan).map(math.log).diff(1)
    mu_1 = log_ret_1.rolling(stats_lookback).mean()
    sigma_1 = log_ret_1.rolling(stats_lookback).std().replace(0, math.nan)

    mu_h = mu_1 * horizon
    sigma_h = sigma_1 * math.sqrt(horizon)

    buy_log_target = math.log1p(max(buy_threshold, 1e-9))
    sell_log_target = math.log1p(max(-0.999999, -abs(sell_threshold)))

    z_up = (buy_log_target - mu_h) / sigma_h
    z_down = (sell_log_target - mu_h) / sigma_h

    cdf_up = normal_cdf_series(z_up)
    cdf_down = normal_cdf_series(z_down)

    out["prob_up_norm_h"] = 1.0 - cdf_up
    out["prob_down_norm_h"] = cdf_down
    out["prob_hold_norm_h"] = (1.0 - out["prob_up_norm_h"] - out["prob_down_norm_h"]).clip(0.0, 1.0)
    out["ret_expected_h"] = mu_h
    out["ret_vol_h"] = sigma_h
    out["signal_to_noise_h"] = mu_h / sigma_h
    out["tail_risk_5pct"] = out["ret_mean_lb"] - 1.65 * out["ret_std_lb"]

    return out


def infer_thresholds(
    close: pd.Series,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    auto_quantile: float | None,
) -> tuple[float, float]:
    if auto_quantile is None:
        return buy_threshold, sell_threshold

    if not (0.5 < auto_quantile < 1.0):
        raise ValueError("--auto-threshold-quantile must be in (0.5, 1.0)")

    fwd = close.shift(-horizon) / close - 1.0
    valid = fwd.dropna()
    if valid.empty:
        raise ValueError("Not enough rows to infer auto thresholds.")

    inferred_buy = float(valid.quantile(auto_quantile))
    inferred_sell = float(valid.quantile(1.0 - auto_quantile))

    buy_out = inferred_buy if inferred_buy > 0 else buy_threshold
    sell_out = abs(inferred_sell) if inferred_sell < 0 else sell_threshold
    return buy_out, sell_out


def merge_higher_timeframe_features(
    base_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    prefix: str,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    stats_lookback: int,
) -> pd.DataFrame:
    htf = add_features(higher_df)
    htf = add_probability_statistics_features(
        htf,
        horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stats_lookback=stats_lookback,
    )
    selected = [
        "ret_1",
        "ret_12",
        "ret_48",
        "ema_ratio_12_26",
        "ema_ratio_12_50",
        "rsi_14",
        "atr_pct_14",
        "bb_width",
        "bb_pos",
        "range_pos_24",
        "prob_up_emp_lb",
        "prob_down_emp_lb",
        "prob_up_norm_h",
        "prob_down_norm_h",
        "ret_mean_lb",
        "ret_std_lb",
        "signal_to_noise_h",
    ]
    selected = [c for c in selected if c in htf.columns]
    rename_map = {c: f"{prefix}{c}" for c in selected}
    htf_features = htf[["date"] + selected].rename(columns=rename_map).sort_values("date")

    base = base_df.sort_values("date")
    merged = pd.merge_asof(base, htf_features, on="date", direction="backward")
    return merged


def prune_redundant_features(
    df: pd.DataFrame,
    base_cols: list[str],
    target_cols: list[str],
    low_variance_epsilon: float,
    corr_threshold: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in base_cols + target_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    low_variance_drop = []
    for col in numeric_cols:
        var = df[col].var(skipna=True)
        if pd.isna(var) or var <= low_variance_epsilon:
            low_variance_drop.append(col)

    kept_numeric = [c for c in numeric_cols if c not in low_variance_drop]
    corr_drop = []
    if 0 < corr_threshold < 1 and len(kept_numeric) > 1:
        corr = df[kept_numeric].corr().abs()
        kept = []
        for col in kept_numeric:
            if any(pd.notna(corr.at[col, prev]) and corr.at[col, prev] >= corr_threshold for prev in kept):
                corr_drop.append(col)
            else:
                kept.append(col)

    drop_cols = low_variance_drop + corr_drop
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, low_variance_drop, corr_drop


def add_labels(df: pd.DataFrame, horizon: int, buy_thresh: float, sell_thresh: float) -> pd.DataFrame:
    out = df.copy()
    out["future_return"] = out["close"].shift(-horizon) / out["close"] - 1.0

    out["label"] = 0
    out.loc[out["future_return"] >= buy_thresh, "label"] = 1
    out.loc[out["future_return"] <= -abs(sell_thresh), "label"] = -1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML dataset from OHLCV")
    parser.add_argument("--input", required=True, help="Input CSV/Feather/Parquet file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--horizon", type=int, default=24, help="Candles ahead for target return")
    parser.add_argument("--buy-threshold", type=float, default=0.004, help="Return threshold for buy label")
    parser.add_argument("--sell-threshold", type=float, default=0.004, help="Return threshold for sell label")
    parser.add_argument("--auto-threshold-quantile", type=float, default=None, help="Auto-infer buy/sell thresholds from future-return quantiles, e.g. 0.7")
    parser.add_argument("--stats-lookback", type=int, default=96, help="Lookback window for statistical/probability features")
    parser.add_argument("--higher-timeframe-input", default=None, help="Optional higher-timeframe OHLCV file to merge via backward asof join")
    parser.add_argument("--higher-timeframe-prefix", default="htf_", help="Prefix for higher-timeframe merged feature columns")
    parser.add_argument("--low-variance-epsilon", type=float, default=1e-12, help="Drop numeric features with variance <= epsilon")
    parser.add_argument("--corr-threshold", type=float, default=0.995, help="Drop redundant numeric features above this absolute correlation")
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if args.stats_lookback < 5:
        raise ValueError("--stats-lookback must be >= 5")
    if not (0 < args.corr_threshold <= 1):
        raise ValueError("--corr-threshold must be in (0, 1]")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_frame(input_path)
    df = normalize_columns(df)
    buy_threshold, sell_threshold = infer_thresholds(
        close=df["close"],
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        auto_quantile=args.auto_threshold_quantile,
    )

    df = add_features(df)
    df = add_probability_statistics_features(
        df,
        horizon=args.horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stats_lookback=args.stats_lookback,
    )

    if args.higher_timeframe_input:
        htf_df = normalize_columns(load_frame(Path(args.higher_timeframe_input)))
        df = merge_higher_timeframe_features(
            base_df=df,
            higher_df=htf_df,
            prefix=args.higher_timeframe_prefix,
            horizon=args.horizon,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            stats_lookback=args.stats_lookback,
        )

    df = add_labels(df, args.horizon, buy_threshold, sell_threshold)

    # Drop rows with rolling/shift NaNs.
    df = df.dropna().reset_index(drop=True)

    base_cols = ["date", "open", "high", "low", "close", "volume"]
    target_cols = ["future_return", "label"]
    df, low_var_drop, corr_drop = prune_redundant_features(
        df=df,
        base_cols=base_cols,
        target_cols=target_cols,
        low_variance_epsilon=args.low_variance_epsilon,
        corr_threshold=args.corr_threshold,
    )

    feature_cols = [c for c in df.columns if c not in base_cols + target_cols]
    keep_cols = base_cols + feature_cols + target_cols
    df = df[keep_cols]
    df.to_csv(output_path, index=False)

    counts = df["label"].value_counts().to_dict()
    label_share = (df["label"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"Wrote {len(df):,} rows to {output_path}")
    print(f"Thresholds used: buy={buy_threshold:.6f}, sell={sell_threshold:.6f}, horizon={args.horizon}")
    print(f"Label counts: {counts}")
    print(f"Label share (%): {label_share}")
    print(f"Dropped low-variance features: {len(low_var_drop)}")
    print(f"Dropped highly-correlated features: {len(corr_drop)}")


if __name__ == "__main__":
    main()
