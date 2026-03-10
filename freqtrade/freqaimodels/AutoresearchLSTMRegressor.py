from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchTransformerTrainer


class _AutoresearchLSTMModel(nn.Module):
    """Windowed multi-feature LSTM regressor with a dense prediction head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        n_layer: int = 3,
        dropout_percent: float = 0.25,
        fc_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout_percent if n_layer > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layer,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(fc_hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, window, features]
        sequence_output, _ = self.lstm(x)
        final_step = sequence_output[:, -1, :]
        y = self.head(final_step)
        # Match WindowDataset target shape: [batch, 1, output_dim]
        return y.unsqueeze(1)


class AutoresearchLSTMRegressor(BasePyTorchRegressor):
    """
    High-capacity LSTM model for FreqAI.

    Uses full engineered feature matrices as rolling sequences and predicts
    labels for the final timestep of each window.
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.window_size < 2:
            raise OperationalException(
                "AutoresearchLSTMRegressor requires freqai.conv_width >= 2. "
                f"Current conv_width={self.window_size}."
            )

        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = float(config.get("learning_rate", 3e-4))
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})

        defaults = {
            "hidden_dim": 512,
            "n_layer": 3,
            "dropout_percent": 0.25,
            "fc_hidden_dim": 256,
        }
        configured = config.get("model_kwargs", {})
        if not isinstance(configured, dict):
            configured = {}
        self.model_kwargs: dict[str, Any] = {**defaults, **configured}

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Fit the LSTM on windowed multivariate features."""

        n_features = data_dictionary["train_features"].shape[-1]
        n_labels = data_dictionary["train_labels"].shape[-1]

        model = _AutoresearchLSTMModel(
            input_dim=n_features,
            output_dim=n_labels,
            **self.model_kwargs,
        )
        model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchTransformerTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                window_size=self.window_size,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )

        trainer.fit(data_dictionary, self.splits)
        return trainer

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """Sliding-window inference with front padding for full-length alignment."""

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            training_filter=False,
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"],
            outlier_check=True,
        )

        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"],
            device=self.device,
        )

        # x shape after unsqueeze: [1, seq_len, features]
        x = x.unsqueeze(0)
        self.model.model.eval()

        yb = torch.empty(0, device=self.device)
        if x.shape[1] > self.window_size:
            ws = self.window_size
            for i in range(0, x.shape[1] - ws):
                xb = x[:, i : i + ws, :].to(self.device)
                y = self.model.model(xb)
                yb = torch.cat((yb, y), dim=1)
        else:
            yb = self.model.model(x)

        yb = yb.cpu().squeeze(0)
        pred_df = pd.DataFrame(yb.detach().numpy(), columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if self.ft_params.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        if x.shape[1] > 1:
            zeros_df = pd.DataFrame(
                np.zeros((x.shape[1] - len(pred_df), len(pred_df.columns))),
                columns=pred_df.columns,
            )
            pred_df = pd.concat([zeros_df, pred_df], axis=0, ignore_index=True)

        return pred_df, dk.do_predict
