# Description: Script with the preprocessing steps for metrics to work
# Author: Anton D. Lautrup
# Date: 16-11-2022

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

#: Policies for categorical evaluation values absent from the train vocabulary.
UNKNOWN_CATEGORY_POLICIES = ("strict", "nominal", "train_mode")


class MixedSchemaPreprocessor:
    """Train-fitted state for role-aware mixed-schema distances.

    Nominal values are deliberately not encoded: equality is evaluated on the
    original values.  Ordinal values are ranked using train vocabulary, while
    continuous values are min-max scaled using train extrema.
    """

    def __init__(self, train_frame, continuous_columns=None, ordinal_columns=None,
                 nominal_columns=None, ordinal_orders=None) -> None:
        self.continuous_columns = list(continuous_columns or [])
        self.ordinal_columns = list(ordinal_columns or [])
        self.nominal_columns = list(nominal_columns or [])
        self.ordinal_orders = ordinal_orders or {}
        columns = self.continuous_columns + self.ordinal_columns + self.nominal_columns
        if not columns:
            raise ValueError("At least one schema column must be provided.")
        missing = [column for column in columns if column not in train_frame.columns]
        if missing:
            raise KeyError(f"Training frame is missing schema columns: {missing}")
        if train_frame.empty:
            raise ValueError("Training population must not be empty.")
        if self.continuous_columns:
            numeric = train_frame[self.continuous_columns]
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError("Continuous training values must be finite and non-missing.")
        self.minimums = train_frame[self.continuous_columns].min() if self.continuous_columns else pd.Series(dtype=float)
        self.maximums = train_frame[self.continuous_columns].max() if self.continuous_columns else pd.Series(dtype=float)
        self.ranks = {}
        for column in self.ordinal_columns:
            if train_frame[column].isna().any():
                raise ValueError(f"Ordinal column {column!r} contains null training values.")
            values = list(self.ordinal_orders.get(column, pd.unique(train_frame[column])))
            if not values:
                raise ValueError(f"Ordinal column {column!r} has no train values.")
            missing = set(train_frame[column]) - set(values)
            if missing:
                raise ValueError(f"Ordinal column {column!r} has values absent from ordinal_orders: {missing}")
            self.ranks[column] = {value: index / max(len(values) - 1, 1) for index, value in enumerate(values)}
        self.fit_role = "train"

    @classmethod
    def fit(cls, train_frame, continuous_columns=None, ordinal_columns=None,
            nominal_columns=None, ordinal_orders=None):
        """Fit mixed-schema transforms using only ``train_frame``."""
        return cls(train_frame, continuous_columns, ordinal_columns, nominal_columns, ordinal_orders)

    def transform(self, frame, role="evaluation"):
        """Return role-separated values without changing fitted state."""
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Frame role={role!r} is missing schema columns: {missing}")
        if frame.empty:
            raise ValueError(f"Population role={role!r} must not be empty.")
        if self.continuous_columns and not np.isfinite(frame[self.continuous_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Continuous values for role={role!r} must be finite and non-missing.")
        result = {
            "continuous": frame[self.continuous_columns].to_numpy(dtype=float) if self.continuous_columns else np.empty((len(frame), 0)),
            "ordinal": np.empty((len(frame), len(self.ordinal_columns)), dtype=float),
            "nominal": frame[self.nominal_columns].to_numpy(dtype=object) if self.nominal_columns else np.empty((len(frame), 0), dtype=object),
        }
        for index, column in enumerate(self.ordinal_columns):
            if frame[column].isna().any():
                raise ValueError(f"Ordinal column {column!r} contains null values for role={role!r}.")
            unknown = set(frame[column]) - set(self.ranks[column])
            if unknown:
                raise ValueError(f"Ordinal column {column!r} contains unseen values for role={role!r}: {unknown}")
            result["ordinal"][:, index] = frame[column].map(self.ranks[column]).to_numpy(dtype=float)
        if self.continuous_columns:
            minimum = self.minimums.to_numpy(dtype=float)
            scale = (self.maximums - self.minimums).replace(0, 1).to_numpy(dtype=float)
            result["continuous"] = (result["continuous"] - minimum) / scale
        return result

    @property
    def columns(self):
        return self.continuous_columns + self.ordinal_columns + self.nominal_columns

    def metadata(self):
        """Return fit-role and train-only state for audit output."""
        return {
            "fit_role": self.fit_role,
            "continuous_columns": list(self.continuous_columns),
            "ordinal_columns": list(self.ordinal_columns),
            "nominal_columns": list(self.nominal_columns),
            "continuous_minimums": self.minimums.tolist(),
            "continuous_maximums": self.maximums.tolist(),
            "ordinal_ranks": {column: {repr(key): value for key, value in ranks.items()} for column, ranks in self.ranks.items()},
        }


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
            # -1 is reserved for explicitly permitted unseen evaluation values.
            # Ordinary transforms remain strict, so synthetic support checks
            # cannot accidentally accept a category absent from train.
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.encoder.fit(train_frame[self.cat_cols])
            # Train mode per column, used to resolve unseen evaluation values
            # under the "train_mode" policy. Ties break on the lowest code so
            # the fitted state is deterministic.
            train_codes = self.encoder.transform(train_frame[self.cat_cols]).astype("int")
            self.fallback_codes = [
                int(np.bincount(train_codes[:, index]).argmax())
                for index in range(len(self.cat_cols))
            ]
        else:
            self.fallback_codes = []
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
            "unknown_category_value": -1 if self.encoder is not None else None,
            "unknown_fallback_codes": list(self.fallback_codes),
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

    def transform(
        self,
        data,
        role="evaluation",
        *,
        allow_unknown_categories=False,
        unknown_policy=None,
    ):
        """Transform a frame without changing fitted encoder/scaler state.

        ``unknown_policy`` controls categorical values absent from train:
        ``"strict"`` raises, ``"nominal"`` keeps the reserved ``-1`` code and
        ``"train_mode"`` replaces it with the column's train-fitted mode.
        ``allow_unknown_categories=True`` is an alias for ``"nominal"``.
        """
        if unknown_policy is None:
            unknown_policy = "nominal" if allow_unknown_categories else "strict"
        if unknown_policy not in UNKNOWN_CATEGORY_POLICIES:
            raise ValueError(f"Unsupported unknown_policy={unknown_policy!r}")
        data = data.copy()
        if self.encoder is not None:
            try:
                encoded_categories = self.encoder.transform(data[self.cat_cols])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Unknown or invalid categorical value while transforming role={role!r}; "
                    f"columns={self.cat_cols}: {exc}"
                ) from exc
            if unknown_policy == "strict" and (encoded_categories == -1).any():
                raise ValueError(
                    f"Unknown categorical value while transforming role={role!r}; "
                    f"columns={self.unknown_categorical_columns(data)}"
                )
            if unknown_policy == "train_mode":
                fallback = np.asarray(self.fallback_codes, dtype=encoded_categories.dtype)
                encoded_categories = np.where(
                    encoded_categories == -1, fallback[np.newaxis, :], encoded_categories
                )
            data[self.cat_cols] = encoded_categories.astype("int")
        if self.num_encoder is not None:
            try:
                data[self.num_cols] = self.num_encoder.transform(data[self.num_cols])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numerical value while transforming role={role!r}; "
                    f"columns={self.num_cols}: {exc}"
                ) from exc
        return data

    def encode(
        self,
        data,
        role="evaluation",
        *,
        allow_unknown_categories=False,
        unknown_policy=None,
    ):
        """Compatibility alias for :meth:`transform`."""
        return self.transform(
            data,
            role=role,
            allow_unknown_categories=allow_unknown_categories,
            unknown_policy=unknown_policy,
        )

    def unknown_row_count(self, data):
        """Return the number of rows with any categorical value outside train support."""
        if self.encoder is None:
            return 0
        try:
            encoded = self.encoder.transform(data[self.cat_cols])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unknown or invalid categorical value while inspecting evaluation data; "
                f"columns={self.cat_cols}: {exc}"
            ) from exc
        return int((encoded == -1).any(axis=1).sum())

    def unknown_categorical_columns(self, data):
        """Return counts of categorical values outside train-fitted support."""
        if self.encoder is None:
            return {}
        try:
            encoded = self.encoder.transform(data[self.cat_cols])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unknown or invalid categorical value while inspecting evaluation data; "
                f"columns={self.cat_cols}: {exc}"
            ) from exc
        return {
            column: int((encoded[:, index] == -1).sum())
            for index, column in enumerate(self.cat_cols)
            if (encoded[:, index] == -1).any()
        }

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
