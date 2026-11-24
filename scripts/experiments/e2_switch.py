#!/usr/bin/env python3
"""
Experiment E2: Empirical Decision-Switch Analysis from Saved Predictions
Protocol: reports/audit/09c_experiment_plan.md
Zero model retraining; strictly post-processing on results/mri_fallback/oof.csv
Development cohort only (N=564), locked test set never touched.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fast_auc(y_true, y_score):
    """Fast Wilcoxon-Mann-Whitney AUC calculation."""
    ranks = rankdata(y_score)
    n1 = np.count_nonzero(y_true == 1)
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT E2: DECISION-SWITCH ANALYSIS (DESCRIPTIVE, ZERO RETRAINING)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load splits and verify locked test exclusion
    sp = pd.read_csv('data/processed/splits/splits_v1.csv')
    test_rids = set(sp[sp.split == 'test'].rid.unique())
    dev_rids = set(sp[sp.split != 'test'].rid.unique())

    # 2. Load cohort and labels
    coh = pd.read_csv('data/processed/cohort_v2b.csv')
    coh = coh[coh.label_primary.notna()].reset_index(drop=True)

    # 3. Load OOF predictions from results/mri_fallback/oof.csv
    oof = pd.read_csv('results/mri_fallback/oof.csv')
    oof_rids = set(oof.rid.unique())

    # Intersection of dev, mri_fallback oof RIDs
    target_rids = sorted(list(dev_rids & oof_rids))
    assert len(target_rids) == 564, f"Expected 564 participants, got {len(target_rids)}"
    assert not (set(target_rids) & test_rids), "CRITICAL: Locked test participant found!"

    coh_dev = coh[coh.RID.isin(target_rids)].drop_duplicates('RID').set_index('RID').reindex(target_rids)
    y = coh_dev['label_primary'].astype(int).values
    phase = coh_dev['adni_phase'].values
    is_adni1 = (phase == 'ADNI1')
    is_later = ~is_adni1

    print(f"Cohort verified: N = {len(target_rids)} participants.", flush=True)
    print(f"Outcome: label_primary (24m), events = {y.sum()} ({y.sum()/len(y)*100:.1f}%).", flush=True)
    print(f"Subgroups: ADNI1 = {is_adni1.sum()}, Later phases = {is_later.sum()}.", flush=True)

    # Pool OOF predictions across repeats (mean prob across repeats)
    models = ['F', 'C', 'C+ROI']
    oof_filt = oof[oof.model.isin(models) & oof.rid.isin(target_rids)]
    pooled = oof_filt.groupby(['rid', 'model'])['prob'].mean().unstack()[models].reindex(target_rids)

    pred_F = pooled['F'].values
    pred_C = pooled['C'].values
    pred_C_ROI = pooled['C+ROI'].values

    # 4. FreeSurfer availability status
    ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(target_rids)
    f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID').reindex(target_rids)
    c29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
    X29 = f29[c29].astype(float)
    fs_init_avail = np.asarray((ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1).values), dtype=bool)

    n_avail = int(fs_init_avail.sum())
    n_unavail = int((~fs_init_avail).sum())
    print(f"FreeSurfer baseline status: {n_avail} available, {n_unavail} unavailable.", flush=True)
    assert n_avail == 544 and n_unavail == 20, f"Expected 544 available / 20 unavailable, got {n_avail}/{n_unavail}"

    # Hippocampal volume / ICV ratio for value-dependent missingness
    hippo_icv = f29['bilateral_hippo_vol_icv'].values
    assert not np.isnan(hippo_icv[fs_init_avail]).any(), "NaN in bilateral_hippo_vol_icv for available FS!"

    # Sort available participant indices by hippo_icv ascending
    avail_indices = np.where(fs_init_avail)[0]
    sorted_avail_indices = avail_indices[np.argsort(hippo_icv[avail_indices])]

    # 5. Missingness simulation
    p_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    n_random_masks = 500
    rng_masks = np.random.default_rng(20261112)

    # Pre-generate random masks as boolean 2D arrays: shape (n_masks, 564)
    random_masks_bool = {}
    for p in p_levels:
        k = int(np.round(p * n_avail))
        masks_arr = np.zeros((n_random_masks, len(target_rids)), dtype=bool)
        for m_idx in range(n_random_masks):
            if k == 0:
                pass
            elif k >= n_avail:
                masks_arr[m_idx, avail_indices] = True
            else:
                chosen = rng_masks.choice(avail_indices, size=k, replace=False)
                masks_arr[m_idx, chosen] = True
        random_masks_bool[p] = masks_arr

    # Value-dependent masks: boolean 1D array of length 564
    val_dep_masks_bool = {}
    for p in p_levels:
        k = int(np.round(p * n_avail))
        mask_arr = np.zeros(len(target_rids), dtype=bool)
        if k > 0:
            mask_arr[sorted_avail_indices[:k]] = True
        val_dep_masks_bool[p] = mask_arr

    # 6. Evaluate Switch Curves
    subgroups = {
        'all': np.ones(len(target_rids), dtype=bool),
        'ADNI1': is_adni1,
        'later': is_later
    }

    curve_records = []
    print("\n--- Computing Switch Curves ---", flush=True)

    for mechanism in ['random', 'value_dependent']:
        for p in p_levels:
            if mechanism == 'random':
                masks_arr = random_masks_bool[p] # shape (500, 564)
                # Available if fs_init_avail and not masked
                fs_avail_matrix = fs_init_avail[None, :] & (~masks_arr) # (500, 564)
                
                # Predictions for all 500 masks
                preds_p1 = np.where(fs_avail_matrix, pred_F[None, :], pred_C_ROI[None, :]) # (500, 564)
                preds_p2 = np.where(fs_avail_matrix, pred_F[None, :], pred_C[None, :])     # (500, 564)

                for sub_name, sub_mask in subgroups.items():
                    sub_y = y[sub_mask]
                    sub_p1 = preds_p1[:, sub_mask]
                    sub_p2 = preds_p2[:, sub_mask]

                    auc_p1 = [fast_auc(sub_y, sub_p1[m]) for m in range(n_random_masks)]
                    auc_p2 = [fast_auc(sub_y, sub_p2[m]) for m in range(n_random_masks)]
                    diffs = [a1 - a2 for a1, a2 in zip(auc_p1, auc_p2)]

                    curve_records.append({
                        'mechanism': mechanism,
                        'subgroup': sub_name,
                        'p_masked': p,
                        'auc_policy1_mean': float(np.mean(auc_p1)),
                        'auc_policy1_std': float(np.std(auc_p1)),
                        'auc_policy2_mean': float(np.mean(auc_p2)),
                        'auc_policy2_std': float(np.std(auc_p2)),
                        'delta_auc_mean': float(np.mean(diffs)),
                        'delta_auc_std': float(np.std(diffs))
                    })
            else: # value_dependent
                mask_arr = val_dep_masks_bool[p]
                fs_curr = fs_init_avail & (~mask_arr)
                pred_p1 = np.where(fs_curr, pred_F, pred_C_ROI)
                pred_p2 = np.where(fs_curr, pred_F, pred_C)

                for sub_name, sub_mask in subgroups.items():
                    sub_y = y[sub_mask]
                    a1 = float(fast_auc(sub_y, pred_p1[sub_mask]))
                    a2 = float(fast_auc(sub_y, pred_p2[sub_mask]))
                    curve_records.append({
                        'mechanism': mechanism,
                        'subgroup': sub_name,
                        'p_masked': p,
                        'auc_policy1_mean': a1,
                        'auc_policy1_std': 0.0,
                        'auc_policy2_mean': a2,
                        'auc_policy2_std': 0.0,
                        'delta_auc_mean': a1 - a2,
                        'delta_auc_std': 0.0
                    })

    df_curve = pd.DataFrame(curve_records)
    df_curve.to_csv('results/experiments/e2_switch_curve.csv', index=False)
    print("Saved results/experiments/e2_switch_curve.csv", flush=True)

    # 7. Paired Participant-Level Bootstrap CIs (2,000 resamples)
    print("\n--- Computing 2,000 Paired Bootstrap CIs at p=0.5 and p=1.0 ---", flush=True)
    ci_records = []
    n_boot = 2000
    rng_boot = np.random.default_rng(20261113)

    for p in [0.5, 1.0]:
        for mechanism in ['random', 'value_dependent']:
            for sub_name, sub_mask in subgroups.items():
                sub_indices = np.where(sub_mask)[0]
                n_sub = len(sub_indices)
                sub_y = y[sub_indices]

                if mechanism == 'value_dependent':
                    mask_arr = val_dep_masks_bool[p]
                    fs_curr = fs_init_avail & (~mask_arr)
                    p1_all = np.where(fs_curr, pred_F, pred_C_ROI)[sub_indices]
                    p2_all = np.where(fs_curr, pred_F, pred_C)[sub_indices]
                    point_est = float(fast_auc(sub_y, p1_all) - fast_auc(sub_y, p2_all))

                    boot_diffs = []
                    for _ in range(n_boot):
                        b_idx = rng_boot.integers(0, n_sub, size=n_sub)
                        yb = sub_y[b_idx]
                        if len(np.unique(yb)) < 2:
                            continue
                        d = fast_auc(yb, p1_all[b_idx]) - fast_auc(yb, p2_all[b_idx])
                        boot_diffs.append(d)
                else: # random
                    # For random, average across 20 representative masks for point and bootstrap
                    masks_arr = random_masks_bool[p][:20] if p < 1.0 else random_masks_bool[p][:1]
                    fs_avail_matrix = fs_init_avail[None, :] & (~masks_arr)
                    p1_m = np.where(fs_avail_matrix, pred_F[None, :], pred_C_ROI[None, :])[:, sub_indices]
                    p2_m = np.where(fs_avail_matrix, pred_F[None, :], pred_C[None, :])[:, sub_indices]

                    point_est = float(np.mean([fast_auc(sub_y, p1_m[i]) - fast_auc(sub_y, p2_m[i]) for i in range(len(masks_arr))]))

                    boot_diffs = []
                    for _ in range(n_boot):
                        b_idx = rng_boot.integers(0, n_sub, size=n_sub)
                        yb = sub_y[b_idx]
                        if len(np.unique(yb)) < 2:
                            continue
                        d = np.mean([fast_auc(yb, p1_m[i, b_idx]) - fast_auc(yb, p2_m[i, b_idx]) for i in range(len(masks_arr))])
                        boot_diffs.append(d)

                ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
                ci_records.append({
                    'mechanism': mechanism,
                    'subgroup': sub_name,
                    'p_masked': p,
                    'delta_auc_point': point_est,
                    'ci_95_lower': float(ci_low),
                    'ci_95_upper': float(ci_high)
                })
                print(f"[{mechanism:15s} | {sub_name:6s} | p={p:.2f}] Delta AUROC = {point_est:+.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])", flush=True)

    df_ci = pd.DataFrame(ci_records)
    df_ci.to_csv('results/experiments/e2_switch_paired_cis.csv', index=False)
    print("Saved results/experiments/e2_switch_paired_cis.csv", flush=True)

    # 8. Publication Figure Generation: F_switch_curve.png
    print("\n--- Generating Figure F_switch_curve.png ---", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    titles = {'all': f'All Usable (N={len(target_rids)})', 'ADNI1': f'ADNI1 Subgroup (N={is_adni1.sum()})', 'later': f'Later Phases (N={is_later.sum()})'}
    
    for ax, sub_name in zip(axes, ['all', 'ADNI1', 'later']):
        sub_c = df_curve[df_curve.subgroup == sub_name]
        
        # Random missingness
        rnd = sub_c[sub_c.mechanism == 'random'].sort_values('p_masked')
        ax.plot(rnd.p_masked, rnd.auc_policy1_mean, 'o-', color='#1f77b4', label='Policy 1: F -> C+ROI (Random)', linewidth=2)
        ax.plot(rnd.p_masked, rnd.auc_policy2_mean, 's--', color='#ff7f0e', label='Policy 2: F -> C (Random)', linewidth=2)
        
        # Value-dependent missingness
        vd = sub_c[sub_c.mechanism == 'value_dependent'].sort_values('p_masked')
        ax.plot(vd.p_masked, vd.auc_policy1_mean, '^:', color='#2ca02c', label='Policy 1: F -> C+ROI (Val-Dep)', linewidth=1.5, alpha=0.85)
        ax.plot(vd.p_masked, vd.auc_policy2_mean, 'v:', color='#d62728', label='Policy 2: F -> C (Val-Dep)', linewidth=1.5, alpha=0.85)

        ax.set_title(titles[sub_name], fontsize=12, fontweight='bold')
        ax.set_xlabel('Missing FreeSurfer Fraction (p)', fontsize=10)
        ax.set_xticks(p_levels)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0.75, 0.92)

    axes[0].set_ylabel('AUROC (Pooled OOF)', fontsize=10)
    axes[0].legend(loc='lower left', fontsize=8, framealpha=0.9)

    plt.suptitle('Experiment E2: Decision-Switch Fallback Trajectory under FreeSurfer Missingness', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/experiments/F_switch_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/experiments/F_switch_curve.png", flush=True)
    print("\nE2 Execution Completed Successfully.", flush=True)

if __name__ == '__main__':
    main()
