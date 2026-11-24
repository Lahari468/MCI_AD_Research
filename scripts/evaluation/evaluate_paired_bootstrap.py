#!/usr/bin/env python3
"""
Paired Bootstrap Comparison Script
Computes participant-level paired differences, 95% empirical percentile CIs, and tail p-values
between two competing model prediction columns.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import run_paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired bootstrap between two prediction models.")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to out-of-fold predictions CSV")
    parser.add_argument("--test_col", type=str, required=True, help="Column name for test model probabilities")
    parser.add_argument("--ref_col", type=str, required=True, help="Column name for reference model probabilities")
    parser.add_argument("--label_col", type=str, default="y_true", help="Ground truth binary label column")
    parser.add_argument("--resamples", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    # If predictions are multi-repeat, average at participant level
    if "RID" in df.columns and df["RID"].duplicated().any():
        df = df.groupby("RID").agg({
            args.label_col: "first",
            args.test_col: "mean",
            args.ref_col: "mean"
        }).reset_index()

    y_true = df[args.label_col].values
    p_test = df[args.test_col].values
    p_ref = df[args.ref_col].values

    results = run_paired_bootstrap(y_true, p_test, p_ref, num_bootstrap=args.resamples, random_seed=args.seed)

    print("\n" + "=" * 60)
    print("PAIRED BOOTSTRAP RESULTS")
    print("=" * 60)
    print(f"Participants Evaluated : {len(y_true)}")
    print(f"Delta AUROC            : {results['delta_auroc']:+.4f} [95% CI: {results['ci_auroc_lo']:+.4f}, {results['ci_auroc_hi']:+.4f}]")
    print(f"Empirical Tail p-value : {results['p_auroc']:.4f}")
    print(f"Delta PR-AUC           : {results['delta_prauc']:+.4f} [95% CI: {results['ci_prauc_lo']:+.4f}, {results['ci_prauc_hi']:+.4f}]")
    print(f"Delta Brier Score      : {results['delta_brier']:+.4f} [95% CI: {results['ci_brier_lo']:+.4f}, {results['ci_brier_hi']:+.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
