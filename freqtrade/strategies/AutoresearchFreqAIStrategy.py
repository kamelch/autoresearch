from __future__ import annotations

from functools import reduce
import math

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class AutoresearchFreqAIStrategy(IStrategy):
    """
    Minimal FreqAI strategy starter for buy/sell with futures short support.

    This is a template to help connect your ML workflow to Freqtrade.
    Tune targets/features/thresholds before any real-money deployment.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count: int = 240
    desired_leverage = 1.0

    # Conservative starter thresholds (predicted future return).
    long_entry_ret = 0.011
    long_exit_ret = -0.001
    short_entry_ret = -0.005
    short_exit_ret = 0.001
    long_prob_min = 0.58
    short_prob_max = 0.44
    long_exit_prob_max = 0.4
    short_exit_prob_min = 0.54

    minimal_roi = {
        "0": 0.02,
        "60": 0.01,
        "180": 0.0,
    }
    stoploss = -0.02
    use_exit_signal = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """Auto-expanded features across config timeframes/periods/shifts/pairs."""
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Basic features expanded across include_timeframes / shifted candles."""
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-log_ret_1"] = dataframe["close"].replace(0, math.nan).map(math.log).diff()
        dataframe["%-ret_std_24"] = dataframe["%-pct-change"].rolling(24).std()
        dataframe["%-ret_mean_24"] = dataframe["%-pct-change"].rolling(24).mean()
        dataframe["%-up_prob_48"] = (dataframe["%-pct-change"] > 0).astype(float).rolling(48).mean()
        dataframe["%-down_prob_48"] = (dataframe["%-pct-change"] < 0).astype(float).rolling(48).mean()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-hl_spread"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"].replace(0, 1)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Non-expanded calendar features."""
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        angle_hour = (2.0 * math.pi * dataframe["%-hour_of_day"]) / 24.0
        angle_dow = (2.0 * math.pi * dataframe["%-day_of_week"]) / 7.0
        dataframe["%-hour_sin"] = angle_hour.map(math.sin)
        dataframe["%-hour_cos"] = angle_hour.map(math.cos)
        dataframe["%-dow_sin"] = angle_dow.map(math.sin)
        dataframe["%-dow_cos"] = angle_dow.map(math.cos)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Regression target: forward return over label_period_candles.

        FreqAI will write predictions back into the same `&-*` column name,
        which we then consume in populate_entry_trend / populate_exit_trend.
        """
        horizon = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        dataframe["&-fwd_return"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        if "&-fwd_return" in dataframe.columns:
            pred = dataframe["&-fwd_return"]
            pred_mean = pred.rolling(96).mean()
            pred_std = pred.rolling(96).std().replace(0, math.nan)
            pred_z = (pred - pred_mean) / pred_std
            sqrt2 = math.sqrt(2.0)
            dataframe["pred_prob_up"] = pred_z.map(
                lambda v: 0.5 * (1.0 + math.erf(v / sqrt2)) if v == v else math.nan
            )
            dataframe["pred_prob_down"] = 1.0 - dataframe["pred_prob_up"]
            dataframe["pred_signal_to_noise"] = pred_mean / pred_std
        else:
            dataframe["pred_prob_up"] = math.nan
            dataframe["pred_prob_down"] = math.nan
            dataframe["pred_signal_to_noise"] = math.nan
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Use conservative leverage by default."""
        return min(self.desired_leverage, max_leverage)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-fwd_return"] > self.long_entry_ret,
            dataframe["pred_prob_up"] >= self.long_prob_min,
            dataframe["volume"] > 0,
        ]
        short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-fwd_return"] < self.short_entry_ret,
            dataframe["pred_prob_up"] <= self.short_prob_max,
            dataframe["volume"] > 0,
        ]
        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "ml_long")
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "ml_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit_conditions = [
            dataframe["do_predict"] == 1,
            (dataframe["&-fwd_return"] < self.long_exit_ret) | (dataframe["pred_prob_up"] <= self.long_exit_prob_max),
            dataframe["volume"] > 0,
        ]
        short_exit_conditions = [
            dataframe["do_predict"] == 1,
            (dataframe["&-fwd_return"] > self.short_exit_ret) | (dataframe["pred_prob_up"] >= self.short_exit_prob_min),
            dataframe["volume"] > 0,
        ]
        if long_exit_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_exit_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "ml_exit")
        if short_exit_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_exit_conditions),
                ["exit_short", "exit_tag"],
            ] = (1, "ml_exit_short")
        return dataframe
