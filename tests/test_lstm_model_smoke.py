from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
FREQTRADE_SRC = ROOT.parent / "freqtrade"
if FREQTRADE_SRC.exists() and str(FREQTRADE_SRC) not in sys.path:
    sys.path.insert(0, str(FREQTRADE_SRC))

MODEL_MODULE = None
MODEL_IMPORT_ERROR: Exception | None = None

try:
    import numpy as np
    import pandas as pd
    import torch

    model_path = ROOT / "freqtrade" / "freqaimodels" / "AutoresearchLSTMRegressor.py"
    spec = importlib.util.spec_from_file_location("autoresearch_lstm_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODEL_MODULE = module
except Exception as exc:  # pragma: no cover - environment-dependent optional deps
    MODEL_IMPORT_ERROR = exc


@unittest.skipUnless(MODEL_MODULE is not None, f"LSTM smoke tests skipped: {MODEL_IMPORT_ERROR!r}")
class LSTMModelSmokeTests(unittest.TestCase):
    def _build_regressor(self):
        reg = object.__new__(MODEL_MODULE.AutoresearchLSTMRegressor)
        reg.window_size = 4
        reg.device = "cpu"
        reg.learning_rate = 1e-3
        reg.trainer_kwargs = {"batch_size": 2, "n_epochs": 1, "n_steps": None}
        reg.model_kwargs = {
            "hidden_dim": 16,
            "n_layer": 1,
            "dropout_percent": 0.0,
            "fc_hidden_dim": 8,
        }
        reg.tb_logger = types.SimpleNamespace(log_scalar=lambda *args, **kwargs: None)
        reg.splits = ["train"]
        reg.ft_params = {"DI_threshold": 0}
        reg.get_init_model = lambda _pair: None
        return reg

    def test_conv_width_guard(self) -> None:
        def fake_base_init(self, **kwargs) -> None:
            self.window_size = 1
            self.freqai_info = {"model_training_parameters": {}}

        with mock.patch.object(MODEL_MODULE.BasePyTorchRegressor, "__init__", fake_base_init):
            with self.assertRaises(MODEL_MODULE.OperationalException):
                MODEL_MODULE.AutoresearchLSTMRegressor()

    def test_fit_and_predict_window_alignment(self) -> None:
        reg = self._build_regressor()

        train_features = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        train_labels = pd.DataFrame(np.random.rand(20, 1), columns=["label"])
        data_dictionary = {
            "train_features": train_features,
            "train_labels": train_labels,
        }

        fit_dk = types.SimpleNamespace(pair="BTC/USDT:USDT")
        trainer = MODEL_MODULE.AutoresearchLSTMRegressor.fit(reg, data_dictionary, fit_dk)
        self.assertIsNotNone(trainer)
        reg.model = trainer

        class DummyFeaturePipeline:
            def transform(self, df, outlier_check=True):
                outliers = np.ones(len(df), dtype=np.int_)
                return df, outliers, None

        class DummyLabelPipeline:
            def inverse_transform(self, df):
                return df, None, None

        class DummyDK:
            def __init__(self):
                self.training_features_list = [f"f{i}" for i in range(5)]
                self.label_list = ["label"]
                self.data_dictionary = {}
                self.feature_pipeline = DummyFeaturePipeline()
                self.label_pipeline = DummyLabelPipeline()
                self.DI_values = None
                self.do_predict = None

            def find_features(self, _df):
                return None

            def filter_features(self, df, features, training_filter=False):
                return df[features], None

        predict_df = train_features.copy()
        predict_dk = DummyDK()
        pred_df, do_predict = MODEL_MODULE.AutoresearchLSTMRegressor.predict(reg, predict_df, predict_dk)

        self.assertEqual(len(pred_df), len(predict_df))
        self.assertEqual(len(do_predict), len(predict_df))
        self.assertTrue((pred_df.iloc[: reg.window_size].to_numpy() == 0).all())
        self.assertFalse(np.isnan(pred_df.to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
