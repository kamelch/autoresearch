from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


@dataclass
class ChronosResidualRidgeModel:
    """Pickle-friendly model object used by FreqAI for inference."""

    regressor: Ridge
    baseline: np.ndarray

    def predict(self, features: Any) -> np.ndarray:
        preds = self.regressor.predict(features)
        preds = np.asarray(preds, dtype=np.float64)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds + self.baseline


class AmazonChronos2Regressor(BaseRegressionModel):
    """
    Hybrid Chronos-2 + Ridge regressor for FreqAI.

    Training flow:
    1) Build a Chronos-2 baseline return from the recent price context.
    2) Fit Ridge on residuals (label - baseline).

    This keeps inference fast and model objects pickle-friendly while still
    enforcing Chronos-2 usage during training windows.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        params = self.model_training_parameters or {}

        self.chronos_model_name = str(params.get("chronos_model", "amazon/chronos-2"))
        self.chronos_context_length = int(params.get("chronos_context_length", 256))
        self.chronos_prediction_length = int(
            params.get(
                "chronos_prediction_length",
                self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 1),
            )
        )
        self.chronos_quantile = float(params.get("chronos_quantile", 0.5))
        self.chronos_device_map = params.get("chronos_device_map", "auto")

        self.price_feature_hint = str(params.get("chronos_price_feature_hint", "raw_price")).lower()
        self.ridge_alpha = float(params.get("ridge_alpha", 1.0))
        self.use_chronos_baseline = bool(params.get("use_chronos_baseline", True))
        self.allow_chronos_fallback = bool(params.get("allow_chronos_fallback", False))

        self._chronos_pipeline: Any | None = None

    def _get_chronos_pipeline(self) -> Any:
        if self._chronos_pipeline is not None:
            return self._chronos_pipeline

        try:
            from chronos import Chronos2Pipeline
        except Exception as exc:
            raise OperationalException(
                "AmazonChronos2Regressor requires the 'chronos-forecasting' package. "
                "Install it in your Freqtrade venv: pip install chronos-forecasting"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.chronos_device_map:
            kwargs["device_map"] = self.chronos_device_map

        try:
            self._chronos_pipeline = Chronos2Pipeline.from_pretrained(self.chronos_model_name, **kwargs)
        except TypeError:
            self._chronos_pipeline = Chronos2Pipeline.from_pretrained(self.chronos_model_name)

        logger.info("Loaded Chronos-2 model: %s", self.chronos_model_name)
        return self._chronos_pipeline

    def _select_price_column(self, feature_df: pd.DataFrame) -> str:
        cols = list(feature_df.columns)
        if not cols:
            raise OperationalException("No feature columns found for Chronos baseline generation.")

        def _find(token: str) -> list[str]:
            token = token.lower()
            return [c for c in cols if token in c.lower()]

        for token in [self.price_feature_hint, "raw_price", "close", "price"]:
            matches = _find(token)
            if matches:
                return matches[0]

        return cols[0]

    def _extract_forecast_value(self, pred_df: pd.DataFrame) -> float:
        if pred_df.empty:
            raise OperationalException("Chronos-2 returned an empty forecast dataframe.")

        if "predictions" in pred_df.columns:
            return float(pred_df["predictions"].iloc[-1])

        if "mean" in pred_df.columns:
            return float(pred_df["mean"].iloc[-1])

        quantile_str = str(self.chronos_quantile)
        if quantile_str in pred_df.columns:
            return float(pred_df[quantile_str].iloc[-1])

        for col in pred_df.columns:
            try:
                col_q = float(str(col))
            except Exception:
                continue
            if abs(col_q - self.chronos_quantile) < 1e-9:
                return float(pred_df[col].iloc[-1])

        numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return float(pred_df[numeric_cols[-1]].iloc[-1])

        raise OperationalException(
            "Chronos-2 forecast dataframe does not contain a usable numeric prediction column."
        )

    def _chronos_baseline_return(self, feature_df: pd.DataFrame) -> float:
        price_col = self._select_price_column(feature_df)
        price_series = pd.to_numeric(feature_df[price_col], errors="coerce").dropna()

        min_points = max(8, self.chronos_prediction_length + 2)
        if len(price_series) < min_points:
            logger.warning("Chronos baseline skipped: not enough usable points in '%s'.", price_col)
            return 0.0

        context = price_series.tail(max(8, self.chronos_context_length))
        last_price = float(context.iloc[-1])
        if abs(last_price) < 1e-9:
            return 0.0

        context_df = pd.DataFrame(
            {
                "id": "series_0",
                "timestamp": pd.date_range(
                    end=pd.Timestamp.utcnow().floor("min"), periods=len(context), freq="min"
                ),
                "target": context.to_numpy(dtype=np.float64),
            }
        )

        pipeline = self._get_chronos_pipeline()
        pred_df = pipeline.predict_df(
            context_df=context_df,
            prediction_length=max(1, self.chronos_prediction_length),
            id_column="id",
            timestamp_column="timestamp",
            target_column="target",
            quantile_levels=[self.chronos_quantile],
        )

        forecast_value = self._extract_forecast_value(pred_df)
        baseline_return = (forecast_value - last_price) / abs(last_price)
        return float(np.clip(baseline_return, -1.0, 1.0))

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        train_weights = data_dictionary.get("train_weights", None)

        if not isinstance(train_features, pd.DataFrame):
            train_features = pd.DataFrame(train_features)
        if not isinstance(train_labels, pd.DataFrame):
            train_labels = pd.DataFrame(train_labels)

        X = train_features.to_numpy(dtype=np.float64)
        y = train_labels.to_numpy(dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        baseline = np.zeros(y.shape[1], dtype=np.float64)
        if self.use_chronos_baseline:
            try:
                baseline_value = self._chronos_baseline_return(train_features)
                baseline[:] = baseline_value
            except Exception as exc:
                if self.allow_chronos_fallback:
                    logger.warning("Chronos baseline failed (%s). Falling back to Ridge-only.", exc)
                else:
                    raise

        y_residual = y - baseline
        model = Ridge(alpha=self.ridge_alpha)
        if train_weights is not None:
            model.fit(X, y_residual, sample_weight=np.asarray(train_weights))
        else:
            model.fit(X, y_residual)

        return ChronosResidualRidgeModel(regressor=model, baseline=baseline)
