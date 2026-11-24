"""Leakage-Safe Preprocessing Pipeline for Multimodal Tabular Features.

Enforces:
  1. fit(df, train_rids): computes statistics (imputation medians, standard scaling mean/std)
     strictly on train_rids. Non-train rows are NEVER accessed or factored into statistics.
  2. transform(df, any_rids): transforms any requested RIDs using the frozen train statistics.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Strictly partitioned imputer and scaler for tabular participant-level features."""

    def __init__(
        self,
        feature_cols: list[str],
        id_col: str = "RID",
        impute_strategy: str = "median",
        scale_strategy: str = "standard",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.id_col = id_col
        self.impute_strategy = impute_strategy
        self.scale_strategy = scale_strategy

        # Fitted statistics
        self.impute_values_: dict[str, float] = {}
        self.scale_mean_: dict[str, float] = {}
        self.scale_std_: dict[str, float] = {}
        self.train_rids_: list[Any] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        train_rids: Sequence[Any],
    ) -> TabularPreprocessor:
        """Fit imputation and scaling statistics strictly on train_rids."""
        train_set = set(train_rids)
        # Isolate training slice strictly by RID
        train_df = df[df[self.id_col].isin(train_set)].copy()

        if train_df.empty:
            raise ValueError("No matching records found for provided train_rids.")

        self.train_rids_ = list(train_rids)
        self.impute_values_ = {}
        self.scale_mean_ = {}
        self.scale_std_ = {}

        # 1. Compute imputation statistics strictly on train slice
        for col in self.feature_cols:
            if col not in train_df.columns:
                raise KeyError(f"Feature column '{col}' not found in dataframe.")
            col_series = pd.to_numeric(train_df[col], errors="coerce")
            valid_vals = col_series.dropna()

            if valid_vals.empty:
                imp_val = 0.0
            elif self.impute_strategy == "median":
                imp_val = float(valid_vals.median())
            elif self.impute_strategy == "mean":
                imp_val = float(valid_vals.mean())
            else:
                raise ValueError(f"Unsupported impute_strategy: {self.impute_strategy}")

            self.impute_values_[col] = imp_val

            # Compute scaling statistics strictly on imputed train slice
            imputed_series = col_series.fillna(imp_val)
            mean_val = float(imputed_series.mean())
            std_val = float(imputed_series.std(ddof=0))
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0

            self.scale_mean_[col] = mean_val
            self.scale_std_[col] = std_val

        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        any_rids: Sequence[Any] | None = None,
    ) -> pd.DataFrame:
        """Transform features for specified RIDs using fitted statistics."""
        if not self.is_fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform.")

        if any_rids is not None:
            target_set = set(any_rids)
            sub_df = df[df[self.id_col].isin(target_set)].copy()
        else:
            sub_df = df.copy()

        out_df = sub_df[[self.id_col]].copy()

        for col in self.feature_cols:
            val_series = pd.to_numeric(sub_df[col], errors="coerce")
            # Apply training imputation value
            imp_val = self.impute_values_[col]
            filled = val_series.fillna(imp_val)

            # Apply training scaling parameters
            if self.scale_strategy == "standard":
                mean_val = self.scale_mean_[col]
                std_val = self.scale_std_[col]
                transformed = (filled - mean_val) / std_val
            elif self.scale_strategy is None or self.scale_strategy == "none":
                transformed = filled
            else:
                raise ValueError(f"Unsupported scale_strategy: {self.scale_strategy}")

            out_df[col] = transformed

        return out_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        train_rids: Sequence[Any],
    ) -> pd.DataFrame:
        """Fit on train_rids and transform train_rids."""
        self.fit(df, train_rids)
        return self.transform(df, train_rids)
