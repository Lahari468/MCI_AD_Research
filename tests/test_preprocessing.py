"""Unit tests for TabularPreprocessor verifying strict data leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import TabularPreprocessor


def test_fit_never_touches_non_train_rows():
    """Verify that mutating non-train rows has zero impact on fitted parameters."""
    data = pd.DataFrame({
        "RID": [1, 2, 3, 4, 5],
        "feature_a": [10.0, 20.0, 30.0, 1000.0, np.nan],
        "feature_b": [1.0, 2.0, 3.0, 9999.0, -500.0],
    })
    train_rids = [1, 2, 3]
    non_train_rids = [4, 5]

    preprocessor = TabularPreprocessor(feature_cols=["feature_a", "feature_b"])
    preprocessor.fit(data, train_rids)

    # Expected values strictly on train slice [10, 20, 30] and [1, 2, 3]
    # Median of [10, 20, 30] = 20.0; Mean = 20.0; Std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3) = sqrt(200/3) = 8.1649658
    assert preprocessor.impute_values_["feature_a"] == 20.0
    assert preprocessor.scale_mean_["feature_a"] == 20.0
    assert np.isclose(preprocessor.scale_std_["feature_a"], np.std([10.0, 20.0, 30.0], ddof=0))

    # Now mutate non-train rows arbitrarily
    data_corrupted = data.copy()
    data_corrupted.loc[data_corrupted["RID"].isin(non_train_rids), "feature_a"] = 99999999.0
    data_corrupted.loc[data_corrupted["RID"].isin(non_train_rids), "feature_b"] = -99999999.0

    preprocessor_corrupted = TabularPreprocessor(feature_cols=["feature_a", "feature_b"])
    preprocessor_corrupted.fit(data_corrupted, train_rids)

    assert preprocessor_corrupted.impute_values_["feature_a"] == preprocessor.impute_values_["feature_a"]
    assert preprocessor_corrupted.impute_values_["feature_b"] == preprocessor.impute_values_["feature_b"]
    assert preprocessor_corrupted.scale_mean_["feature_a"] == preprocessor.scale_mean_["feature_a"]
    assert preprocessor_corrupted.scale_mean_["feature_b"] == preprocessor.scale_mean_["feature_b"]
    assert preprocessor_corrupted.scale_std_["feature_a"] == preprocessor.scale_std_["feature_a"]
    assert preprocessor_corrupted.scale_std_["feature_b"] == preprocessor.scale_std_["feature_b"]


def test_transform_uses_fitted_parameters():
    """Verify transform applies the frozen fitted imputer and scaler."""
    data = pd.DataFrame({
        "RID": [1, 2, 3, 4],
        "feat": [10.0, 20.0, 30.0, np.nan],
    })
    train_rids = [1, 2, 3]

    preprocessor = TabularPreprocessor(feature_cols=["feat"])
    preprocessor.fit(data, train_rids)

    # Transform non-train participant with missing value
    transformed = preprocessor.transform(data, any_rids=[4])
    # Participant 4 has np.nan -> imputed with median (20.0) -> scaled: (20 - 20) / std = 0.0
    val = transformed.loc[transformed["RID"] == 4, "feat"].iloc[0]
    assert np.isclose(val, 0.0)


def test_transform_before_fit_raises():
    """Verify transform raises RuntimeError if called prior to fit."""
    data = pd.DataFrame({"RID": [1], "feat": [10.0]})
    preprocessor = TabularPreprocessor(feature_cols=["feat"])
    with pytest.raises(RuntimeError):
        preprocessor.transform(data)
