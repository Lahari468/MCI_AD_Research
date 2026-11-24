#!/usr/bin/env python3
"""Stage 9 Baseline Modeling Pipeline.

Enforces:
  1. Dev set only evaluation using 3x5 Repeated Stratified CV from splits_v1.csv.
  2. Models: L2 Logistic Regression and HistGradientBoosting (inner CV on training folds only).
  3. Feature sets: phase-only, clinical_full, clinical_nocirc, freesurfer,
     clinical_full+freesurfer, clinical_nocirc+freesurfer, with and without adni_phase.
  4. Metrics: AUROC, PR-AUC, Brier score, Log Loss, Balanced Accuracy,
     with 1,000 bootstrap 95% CI resampled at the RID level.
  5. Per-phase group evaluations (ADNI1, ADNIGO+ADNI2, ADNI3) for the best model of each feature set.
  6. Leave-One-Phase-Group-Out (LOPGO) domain generalization.
  7. Artifact exports to results/baselines/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Feature set definitions
CLINICAL_FULL_COLS = [
    "age",
    "sex",
    "PTEDUCAT",
    "APOE4",
    "MMSCORE",
    "CDRSB",
    "CDGLOBAL",
    "FAQTOTAL",
    "TOTSCORE",
    "TOTAL13",
]

CLINICAL_NOCIRC_COLS = [
    "age",
    "sex",
    "PTEDUCAT",
    "APOE4",
    "MMSCORE",
    "TOTSCORE",
    "TOTAL13",
]

FREESURFER_COLS = [
    "ST29SV",
    "ST88SV",
    "ST24TA",
    "ST83TA",
    "ST44TA",
    "ST103TA",
    "ST40TA",
    "ST99TA",
    "ST52TA",
    "ST111TA",
    "ST10CV",
    "left_hippo_vol_icv",
    "right_hippo_vol_icv",
    "bilateral_hippo_vol_icv",
]

FS29_COLS = [
    # 8 subcortical volumes
    "ST29SV",
    "ST88SV",
    "ST12SV",
    "ST71SV",
    "ST37SV",
    "ST96SV",
    "ST30SV",
    "ST89SV",
    # 20 cortical thickness averages
    "ST24TA",
    "ST83TA",
    "ST44TA",
    "ST103TA",
    "ST26TA",
    "ST85TA",
    "ST32TA",
    "ST91TA",
    "ST40TA",
    "ST99TA",
    "ST52TA",
    "ST111TA",
    "ST50TA",
    "ST109TA",
    "ST34TA",
    "ST93TA",
    "ST31TA",
    "ST90TA",
    "ST58TA",
    "ST117TA",
    # 1 intracranial volume
    "ST10CV",
]

PHASE_INDICATOR_COLS = [
    "phase_ADNI1",
    "phase_ADNIGO",
    "phase_ADNI2",
    "phase_ADNI3",
]


def load_dev_data(
    splits_path: str = "data/processed/splits/splits_v1.csv",
    cohort_path: str = "data/processed/cohort_v2.csv",
    clinical_path: str = "data/processed/features/clinical_raw_v1.csv",
    fs_path: str = "data/processed/features/freesurfer_raw_v1.csv",
    fs29_path: str = "data/processed/features/freesurfer29_raw_v1.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge dev-set features and cross-validation splits."""
    sp = pd.read_csv(splits_path)
    dev_sp = sp[sp["split"] == "dev"].copy()
    dev_rids = set(dev_sp["rid"].unique())

    c2 = pd.read_csv(cohort_path)
    clin = pd.read_csv(clinical_path)
    fs = pd.read_csv(fs_path)

    # Base cohort dataframe
    df = c2[c2["RID"].isin(dev_rids)][["RID", "label_primary", "adni_phase"]].copy()
    df["label_primary"] = df["label_primary"].astype(int)

    # Merge clinical & FreeSurfer features
    df = pd.merge(df, clin, on="RID", how="left")
    df = pd.merge(df, fs, on="RID", how="left")

    # Merge fs29 features if available
    if Path(fs29_path).exists():
        fs29 = pd.read_csv(fs29_path)
        new_cols = ["RID"] + [c for c in fs29.columns if c not in df.columns and c != "RID"]
        df = pd.merge(df, fs29[new_cols], on="RID", how="left")

    # One-hot encode phase
    for p in ["ADNI1", "ADNIGO", "ADNI2", "ADNI3"]:
        df[f"phase_{p}"] = (df["adni_phase"] == p).astype(float)

    # Phase groups: G1=ADNI1, G2=ADNIGO+ADNI2, G3=ADNI3
    def assign_group(ph: str) -> str:
        if ph == "ADNI1":
            return "ADNI1"
        elif ph in ["ADNIGO", "ADNI2"]:
            return "ADNIGO+ADNI2"
        elif ph == "ADNI3":
            return "ADNI3"
        return "Other"

    df["phase_group"] = df["adni_phase"].apply(assign_group)
    return df, dev_sp


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard classification evaluation metrics."""
    eps = 1e-7
    prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, prob_clipped)
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "log_loss": float(ll),
        "balanced_acc": float(bacc),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for classification metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_stats: dict[str, list[float]] = {
        "auroc": [],
        "pr_auc": [],
        "brier": [],
        "log_loss": [],
        "balanced_acc": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], threshold=threshold)
        for k in boot_stats:
            if not np.isnan(m[k]):
                boot_stats[k].append(m[k])

    ci_dict = {}
    for k in boot_stats:
        vals = boot_stats[k]
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
        else:
            low, high = np.nan, np.nan
        ci_dict[k] = (low, high)
    return ci_dict


def find_optimal_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Find decision threshold maximizing balanced accuracy on validation slice."""
    if grid is None:
        grid = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.5
    best_score = -1.0
    for t in grid:
        score = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        if score > best_score:
            best_score = score
            best_thresh = float(t)
    return best_thresh


def build_feature_dictionary() -> dict[str, list[str]]:
    """Construct feature column lists for all evaluated configurations."""
    configs = {
        "phase_only": PHASE_INDICATOR_COLS,
        "clinical_full": CLINICAL_FULL_COLS,
        "clinical_full_with_phase": CLINICAL_FULL_COLS + PHASE_INDICATOR_COLS,
        "clinical_nocirc": CLINICAL_NOCIRC_COLS,
        "clinical_nocirc_with_phase": CLINICAL_NOCIRC_COLS + PHASE_INDICATOR_COLS,
        "freesurfer": FREESURFER_COLS,
        "freesurfer_with_phase": FREESURFER_COLS + PHASE_INDICATOR_COLS,
        "clinical_full_plus_freesurfer": CLINICAL_FULL_COLS + FREESURFER_COLS,
        "clinical_full_plus_freesurfer_with_phase": CLINICAL_FULL_COLS + FREESURFER_COLS + PHASE_INDICATOR_COLS,
        "clinical_nocirc_plus_freesurfer": CLINICAL_NOCIRC_COLS + FREESURFER_COLS,
        "clinical_nocirc_plus_freesurfer_with_phase": CLINICAL_NOCIRC_COLS + FREESURFER_COLS + PHASE_INDICATOR_COLS,
    }
    return configs


def get_model_grid(model_type: str) -> tuple[Any, dict[str, list[Any]]]:
    """Return model estimator and hyperparameter grid for inner CV."""
    if model_type == "lr":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
        ])
        grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }
    elif model_type == "hgb":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ])
        grid = {
            "clf__max_iter": [50, 100],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
            "clf__min_samples_leaf": [10, 20],
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return estimator, grid


def run_3x5_cv(
    df: pd.DataFrame,
    dev_sp: pd.DataFrame,
    feature_dict: dict[str, list[str]],
    model_types: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute 3x5 CV over all model and feature set configurations."""
    rids_ordered = df["RID"].values
    y_true_all = df["label_primary"].values

    # Storage for out-of-fold predictions per RID and config
    oof_preds_dict: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    if model_types is None:
        model_types = [("lr", "LogisticRegression_L2"), ("hgb", "HistGradientBoosting")]

    total_configs = len(feature_dict) * len(model_types)
    cfg_idx = 0

    for model_code, model_name in model_types:
        for feat_name, cols in feature_dict.items():
            cfg_idx += 1
            config_key = f"{model_code}_{feat_name}"
            print(f"[{cfg_idx}/{total_configs}] Running 3x5 CV for {model_name} on {feat_name} ({len(cols)} features)...")

            X_mat = df[cols].values

            # Accumulate out-of-fold predictions across repeats
            repeat_preds = np.zeros((3, len(df)), dtype=float)
            fold_thresholds: list[float] = []

            for rep in range(3):
                rep_sp = dev_sp[dev_sp["repeat"] == rep]
                for f in range(5):
                    val_rids = set(rep_sp[rep_sp["fold"] == f]["rid"])
                    val_mask = df["RID"].isin(val_rids).values
                    train_mask = ~val_mask

                    X_train, y_train = X_mat[train_mask], y_true_all[train_mask]
                    X_val, y_val = X_mat[val_mask], y_true_all[val_mask]

                    estimator, param_grid = get_model_grid(model_code)
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + rep * 10 + f)
                    clf = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=inner_cv,
                        scoring="roc_auc",
                        n_jobs=-1,
                    )
                    clf.fit(X_train, y_train)

                    p_val = clf.predict_proba(X_val)[:, 1]
                    repeat_preds[rep, val_mask] = p_val

                    # Threshold optimization strictly on validation fold
                    thresh = find_optimal_threshold(y_val, p_val)
                    fold_thresholds.append(thresh)

            # Pooled out-of-fold prediction per RID (averaged over 3 repeats)
            p_oof = repeat_preds.mean(axis=0)
            oof_preds_dict[config_key] = p_oof

            mean_threshold = float(np.mean(fold_thresholds))
            metrics = compute_metrics(y_true_all, p_oof, threshold=mean_threshold)
            ci = bootstrap_metrics(y_true_all, p_oof, threshold=mean_threshold, n_bootstraps=1000)

            summary_rows.append({
                "model_type": model_name,
                "feature_set": feat_name,
                "has_adni_phase": "with_phase" in feat_name or feat_name == "phase_only",
                "num_features": len(cols),
                "threshold": round(mean_threshold, 4),
                "auroc": round(metrics["auroc"], 4),
                "auroc_ci_low": round(ci["auroc"][0], 4),
                "auroc_ci_high": round(ci["auroc"][1], 4),
                "pr_auc": round(metrics["pr_auc"], 4),
                "pr_auc_ci_low": round(ci["pr_auc"][0], 4),
                "pr_auc_ci_high": round(ci["pr_auc"][1], 4),
                "brier": round(metrics["brier"], 4),
                "brier_ci_low": round(ci["brier"][0], 4),
                "brier_ci_high": round(ci["brier"][1], 4),
                "log_loss": round(metrics["log_loss"], 4),
                "log_loss_ci_low": round(ci["log_loss"][0], 4),
                "log_loss_ci_high": round(ci["log_loss"][1], 4),
                "balanced_acc": round(metrics["balanced_acc"], 4),
                "balanced_acc_ci_low": round(ci["balanced_acc"][0], 4),
                "balanced_acc_ci_high": round(ci["balanced_acc"][1], 4),
            })

    # Assemble OOF dataframe
    oof_df = pd.DataFrame({
        "RID": df["RID"],
        "label_primary": df["label_primary"],
        "adni_phase": df["adni_phase"],
        "phase_group": df["phase_group"],
    })
    for k, preds in oof_preds_dict.items():
        oof_df[f"prob_{k}"] = preds

    summary_df = pd.DataFrame(summary_rows)
    return oof_df, summary_df


def run_per_phase_metrics(
    oof_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate per-phase metrics for the best model of each feature set."""
    base_feat_sets = [
        "phase_only",
        "clinical_full",
        "clinical_nocirc",
        "freesurfer",
        "clinical_full_plus_freesurfer",
        "clinical_nocirc_plus_freesurfer",
    ]

    phase_rows: list[dict[str, Any]] = []

    for base_feat in base_feat_sets:
        # Find all variants for this base feature set
        sub_sum = summary_df[summary_df["feature_set"].str.startswith(base_feat)].sort_values("auroc", ascending=False)
        best_row = sub_sum.iloc[0]
        model_type_code = "lr" if "LogisticRegression" in best_row["model_type"] else "hgb"
        prob_col = f"prob_{model_type_code}_{best_row['feature_set']}"
        threshold = best_row["threshold"]

        # Evaluate across the 3 phase groups
        for p_grp in ["ADNI1", "ADNIGO+ADNI2", "ADNI3"]:
            sub_grp = oof_df[oof_df["phase_group"] == p_grp]
            y_g = sub_grp["label_primary"].values
            p_g = sub_grp[prob_col].values
            m = compute_metrics(y_g, p_g, threshold=threshold)

            phase_rows.append({
                "base_feature_set": base_feat,
                "best_model_config": f"{best_row['model_type']} ({best_row['feature_set']})",
                "phase_group": p_grp,
                "n_samples": len(sub_grp),
                "n_positives": int(y_g.sum()),
                "prevalence": round(float(y_g.mean()), 4),
                "auroc": round(m["auroc"], 4),
                "pr_auc": round(m["pr_auc"], 4),
                "brier": round(m["brier"], 4),
                "log_loss": round(m["log_loss"], 4),
                "balanced_acc": round(m["balanced_acc"], 4),
            })

    return pd.DataFrame(phase_rows)


def run_leave_one_phase_group_out(
    df: pd.DataFrame,
    feature_dict: dict[str, list[str]],
) -> pd.DataFrame:
    """Execute Leave-One-Phase-Group-Out (LOPGO) domain generalization."""
    print("\nExecuting Leave-One-Phase-Group-Out (LOPGO) domain generalization...")
    groups = ["ADNI1", "ADNIGO+ADNI2", "ADNI3"]
    lopgo_rows: list[dict[str, Any]] = []

    model_types = [("lr", "LogisticRegression_L2"), ("hgb", "HistGradientBoosting")]

    for test_grp in groups:
        train_mask = (df["phase_group"] != test_grp).values
        test_mask = (df["phase_group"] == test_grp).values

        y_train = df.loc[train_mask, "label_primary"].values
        y_test = df.loc[test_mask, "label_primary"].values

        for model_code, model_name in model_types:
            for feat_name, cols in feature_dict.items():
                X_train = df.loc[train_mask, cols].values
                X_test = df.loc[test_mask, cols].values

                estimator, param_grid = get_model_grid(model_code)
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                clf = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                clf.fit(X_train, y_train)

                p_test = clf.predict_proba(X_test)[:, 1]
                # Default threshold 0.5 for out-of-domain evaluation
                m = compute_metrics(y_test, p_test, threshold=0.5)

                lopgo_rows.append({
                    "test_group": test_grp,
                    "train_groups": "+".join([g for g in groups if g != test_grp]),
                    "model_type": model_name,
                    "feature_set": feat_name,
                    "n_test": len(y_test),
                    "n_pos_test": int(y_test.sum()),
                    "auroc": round(m["auroc"], 4),
                    "pr_auc": round(m["pr_auc"], 4),
                    "brier": round(m["brier"], 4),
                    "log_loss": round(m["log_loss"], 4),
                    "balanced_acc": round(m["balanced_acc"], 4),
                })

    return pd.DataFrame(lopgo_rows)


def run_fs29_evaluation(
    summary_path: str = "results/baselines/summary.csv",
) -> pd.DataFrame:
    """Run ONLY the 3 requested fs29 configs with L2 logistic regression and append to summary.csv."""
    df, dev_sp = load_dev_data()
    print(f"Loaded {len(df)} dev participants for fs29 baseline evaluation.")

    fs29_configs = {
        "fs29_with_phase": FS29_COLS + PHASE_INDICATOR_COLS,
        "clinical_full_plus_fs29_with_phase": CLINICAL_FULL_COLS + FS29_COLS + PHASE_INDICATOR_COLS,
        "clinical_nocirc_plus_fs29_with_phase": CLINICAL_NOCIRC_COLS + FS29_COLS + PHASE_INDICATOR_COLS,
    }

    # Run ONLY L2 logistic regression on the three configs
    oof_df, fs29_summary_df = run_3x5_cv(
        df=df,
        dev_sp=dev_sp,
        feature_dict=fs29_configs,
        model_types=[("lr", "LogisticRegression_L2")],
    )

    # Append to results/baselines/summary.csv without overwriting
    sum_file = Path(summary_path)
    if sum_file.exists():
        existing_summary = pd.read_csv(sum_file)
        filtered_existing = existing_summary[
            ~((existing_summary["model_type"] == "LogisticRegression_L2")
              & (existing_summary["feature_set"].isin(fs29_configs.keys())))
        ]
        combined_summary = pd.concat([filtered_existing, fs29_summary_df], ignore_index=True)
    else:
        combined_summary = fs29_summary_df

    combined_summary.to_csv(sum_file, index=False)
    print(f"\nAppended {len(fs29_summary_df)} fs29 configurations to {sum_file} (total rows: {len(combined_summary)}).")

    print("\n=== FS29 BASELINE EVALUATION (L2 Logistic Regression with ADNI Phase) ===")
    for _, r in fs29_summary_df.iterrows():
        print(f"[{r['model_type']}] {r['feature_set']}: AUROC = {r['auroc']:.4f} (95% CI: {r['auroc_ci_low']:.4f}-{r['auroc_ci_high']:.4f}) | PR-AUC = {r['pr_auc']:.4f} (95% CI: {r['pr_auc_ci_low']:.4f}-{r['pr_auc_ci_high']:.4f})")

    return fs29_summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline model training and evaluation.")
    parser.add_argument("--fs29", action="store_true", help="Run only the 3 a-priori fs29 configurations with L2 Logistic Regression.")
    args = parser.parse_args()

    if args.fs29:
        run_fs29_evaluation()
        return

    out_dir = Path("results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dev set features and split allocations...")
    df, dev_sp = load_dev_data()
    print(f"Loaded {len(df)} dev participants across {df['phase_group'].nunique()} phase groups.")

    feature_dict = build_feature_dictionary()

    # 1. 3x5 CV Evaluation
    oof_df, summary_df = run_3x5_cv(df=df, dev_sp=dev_sp, feature_dict=feature_dict)

    # Export OOF predictions
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved out-of-fold predictions to {oof_path} ({oof_df.shape}).")

    # Export 3x5 CV summary
    sum_path = out_dir / "summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved 3x5 CV summary metrics to {sum_path} ({len(summary_df)} rows).")

    # 2. Per-phase metrics for best model of each feature set
    phase_metrics_df = run_per_phase_metrics(oof_df=oof_df, summary_df=summary_df)
    phase_metrics_path = out_dir / "phase_stratified_summary.csv"
    phase_metrics_df.to_csv(phase_metrics_path, index=False)
    print(f"Saved phase-stratified metrics to {phase_metrics_path}.")

    # 3. Leave-one-phase-group-out (LOPGO)
    lopgo_df = run_leave_one_phase_group_out(df=df, feature_dict=feature_dict)
    lopgo_path = out_dir / "leave_one_phase_group_out.csv"
    lopgo_df.to_csv(lopgo_path, index=False)
    print(f"Saved LOPGO evaluation to {lopgo_path}.")

    print("\n=== TOP 5 BASELINE MODELS (by AUROC) ===")
    top5 = summary_df.sort_values("auroc", ascending=False).head(5)
    print(top5[["model_type", "feature_set", "auroc", "pr_auc", "brier", "balanced_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()
