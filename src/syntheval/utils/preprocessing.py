# Description: Script with the preprocessing steps for metrics to work
# Author: Anton D. Lautrup
# Date: 16-11-2022

import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


class TrainFittedPreprocessor:
    """Encode evaluation frames using state fitted on one real train frame."""

    def __init__(self, train_frame, categorical_columns, numerical_columns) -> None:
        self.cat_cols = list(categorical_columns or [])
        self.num_cols = list(numerical_columns or [])
        if not self.cat_cols and not self.num_cols:
            raise ValueError("Either categorical or numerical columns must be provided.")

        missing = [
            column
            for column in self.cat_cols + self.num_cols
            if column not in train_frame.columns
        ]
        if missing:
            raise KeyError(f"Training frame is missing preprocessing columns: {missing}")

        self.encoder = None
        self.num_encoder = None
        if self.cat_cols:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(train_frame[self.cat_cols])
        if self.num_cols:
            self.num_encoder = MinMaxScaler()
            self.num_encoder.fit(train_frame[self.num_cols])
        self.fit_role = "train"
        self.fingerprint = self._build_fingerprint()

    @classmethod
    def fit(cls, train_frame, categorical_columns, numerical_columns):
        """Fit preprocessing state using only ``train_frame``."""
        return cls(train_frame, categorical_columns, numerical_columns)

    def _build_fingerprint(self) -> str:
        payload = self._state_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _state_payload(self):
        return {
            "fit_role": self.fit_role,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "categories": (
                [[repr(value) for value in values] for values in self.encoder.categories_]
                if self.encoder is not None
                else None
            ),
            "data_min": self.num_encoder.data_min_.tolist() if self.num_encoder is not None else None,
            "data_max": self.num_encoder.data_max_.tolist() if self.num_encoder is not None else None,
            "data_range": (
                self.num_encoder.data_range_.tolist() if self.num_encoder is not None else None
            ),
            "scale": self.num_encoder.scale_.tolist() if self.num_encoder is not None else None,
            "offset": self.num_encoder.min_.tolist() if self.num_encoder is not None else None,
            "feature_range": (
                list(self.num_encoder.feature_range) if self.num_encoder is not None else None
            ),
        }

    def transform(self, data, role="evaluation"):
        """Transform a frame without changing fitted encoder/scaler state."""
        data = data.copy()
        if self.encoder is not None:
            try:
                data[self.cat_cols] = self.encoder.transform(data[self.cat_cols]).astype("int")
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Unknown or invalid categorical value while transforming role={role!r}; "
                    f"columns={self.cat_cols}: {exc}"
                ) from exc
        if self.num_encoder is not None:
            try:
                data[self.num_cols] = self.num_encoder.transform(data[self.num_cols])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numerical value while transforming role={role!r}; "
                    f"columns={self.num_cols}: {exc}"
                ) from exc
        return data

    def encode(self, data, role="evaluation"):
        """Compatibility alias for :meth:`transform`."""
        return self.transform(data, role=role)

    def decode(self, data):
        """Decode a transformed frame using the fitted train vocabulary/range."""
        data = data.copy()
        if self.encoder is not None:
            data[self.cat_cols] = self.encoder.inverse_transform(data[self.cat_cols])
        if self.num_encoder is not None:
            data[self.num_cols] = self.num_encoder.inverse_transform(data[self.num_cols])
        return data

    def metadata(self):
        """Return serializable preprocessing provenance."""
        metadata = self._state_payload()
        metadata["categorical_columns"] = list(self.cat_cols)
        metadata["numerical_columns"] = list(self.num_cols)
        metadata["fingerprint"] = self.fingerprint
        return metadata


def stack(real, fake):
    """Function for stacking the real and fake dataframes and adding a column for keeping
    track of which is which. This is essentially to ease the use of seaborn plots hue.
    """
    real = pd.concat(
        (real.reset_index(), pd.DataFrame(np.ones(len(real)), columns=["real"])), axis=1
    )
    fake = pd.concat(
        (fake.reset_index(), pd.DataFrame(np.zeros(len(fake)), columns=["real"])),
        axis=1,
    )
    return pd.concat((real, fake), ignore_index=True)


class consistent_label_encoding:
    """Legacy pooled encoder retained for historical evaluation paths.

    New SynthEval evaluations use :class:`TrainFittedPreprocessor` so candidate
    and holdout values cannot influence fitted transform state.
    """

    def __init__(
        self, real, fake, categorical_columns, numerical_columns, hout=None
    ) -> None:
        assert (
            len(categorical_columns) > 0 or len(numerical_columns) > 0
        ), "Either categorical or nummerical columns must be provided."

        joint_dataframe = pd.concat((real.reset_index(), fake.reset_index()), axis=0)
        if hout is not None:
            joint_dataframe = pd.concat(
                (joint_dataframe.reset_index(), hout.reset_index()), axis=0
            )

        if len(categorical_columns) > 0:
            self.encoder = OrdinalEncoder().fit(joint_dataframe[categorical_columns])
            self.cat_cols = categorical_columns
        else:
            self.cat_cols = None

        if len(numerical_columns) > 0:
            self.num_encoder = MinMaxScaler().fit(joint_dataframe[numerical_columns])
            self.num_cols = numerical_columns
        else:
            self.num_cols = None
        pass

    def encode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.transform(data[self.cat_cols]).astype(
                "int"
            )
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.transform(data[self.num_cols])
        return data

    def decode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.inverse_transform(data[self.cat_cols])
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.inverse_transform(data[self.num_cols])
        return data