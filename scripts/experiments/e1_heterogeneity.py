"""
scripts/experiments/e1_heterogeneity.py

Post hoc descriptive evaluation of subgroup heterogeneity and statistical precision
from saved E1 predictions (results/experiments/e1_oof.csv).
NO MODEL FITTING. Outcome: label_primary (24m).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def fast_auc(y_true, y_score):
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(y_score)
    return (r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def compute_heterogeneity():
    oof_path = 'results/experiments/e1_oof.csv'
    coh_path = 'data/processed/cohort_v2b.csv'
    out_csv = 'results/experiments/e1_heterogeneity.csv'

    assert os.path.exists(oof_path), f"Missing {oof_path}"
    assert os.path.exists(coh_path), f"Missing {coh_path}"

    oof = pd.read_csv(oof_path)
    coh = pd.read_csv(coh_path).drop_duplicates('RID').set_index('RID')

    rids = np.array(sorted(oof['rid'].unique()))
    assert len(rids) == 564, f"Expected 564 participants, got {len(rids)}"

    y = coh.reindex(rids)['label_primary'].astype(int).values
    phase = coh.reindex(rids)['adni_phase'].values
    is_adni1 = (phase == 'ADNI1')

    idx_adni1 = np.where(is_adni1)[0]
    idx_later = np.where(~is_adni1)[0]

    assert len(idx_adni1) == 188 and y[idx_adni1].sum() == 72
    assert len(idx_later) == 376 and y[idx_later].sum() == 51

    # Average predicted probabilities across 10 repeats
    pooled_prob = {}
    for m in oof['model'].unique():
        sub = oof[oof.model == m].sort_values(['rid', 'repeat'])
        pooled_prob[m] = sub.groupby('rid')['prob'].mean().reindex(rids).values

    contrasts = [
        ('C+ROI18 - C', 'C+ROI18', 'C'),
        ('C+CTRL18 - C', 'C+CTRL18', 'C'),
        ('C+ROI18 - C+CTRL18', 'C+ROI18', 'C+CTRL18'),
        ('C+ph+ROI18 - C+ph', 'C+ph+ROI18', 'C+ph'),
        ('C+prot+ROI18 - C+prot', 'C+prot+ROI18', 'C+prot'),
        ('F - C', 'F', 'C'),
        ('F+ROI18 - F', 'F+ROI18', 'F')
    ]

    B = 2000
    rng = np.random.RandomState(20261114)
    boot_adni1_idx = [rng.choice(idx_adni1, size=len(idx_adni1), replace=True) for _ in range(B)]
    boot_later_idx = [rng.choice(idx_later, size=len(idx_later), replace=True) for _ in range(B)]

    records = []
    print("Computing subgroup heterogeneity across 7 contrasts (B=2000)...")

    for label, comp_m, ref_m in contrasts:
        p_comp = pooled_prob[comp_m]
        p_ref = pooled_prob[ref_m]

        # Point estimates
        pool_gain = fast_auc(y, p_comp) - fast_auc(y, p_ref)
        adni1_gain = fast_auc(y[idx_adni1], p_comp[idx_adni1]) - fast_auc(y[idx_adni1], p_ref[idx_adni1])
        later_gain = fast_auc(y[idx_later], p_comp[idx_later]) - fast_auc(y[idx_later], p_ref[idx_later])
        diff_gain = adni1_gain - later_gain

        boot_adni1 = np.empty(B)
        boot_later = np.empty(B)
        boot_diff = np.empty(B)
        boot_pool = np.empty(B)

        for b in range(B):
            b_a = boot_adni1_idx[b]
            b_l = boot_later_idx[b]

            y_a = y[b_a]
            y_l = y[b_l]

            auc_a = fast_auc(y_a, p_comp[b_a]) - fast_auc(y_a, p_ref[b_a])
            auc_l = fast_auc(y_l, p_comp[b_l]) - fast_auc(y_l, p_ref[b_l])

            boot_adni1[b] = auc_a
            boot_later[b] = auc_l
            boot_diff[b] = auc_a - auc_l

            b_pool = np.concatenate([b_a, b_l])
            boot_pool[b] = fast_auc(y[b_pool], p_comp[b_pool]) - fast_auc(y[b_pool], p_ref[b_pool])

        se_pool = float(np.std(boot_pool, ddof=1))
        mdd = 2.8 * se_pool

        ci_adni1 = np.percentile(boot_adni1, [2.5, 97.5])
        ci_later = np.percentile(boot_later, [2.5, 97.5])
        ci_diff = np.percentile(boot_diff, [2.5, 97.5])

        records.append({
            'contrast': label,
            'pooled_gain': round(float(pool_gain), 4),
            'pooled_se': round(se_pool, 4),
            'mdd_2p8_se': round(float(mdd), 4),
            'adni1_gain': round(float(adni1_gain), 4),
            'adni1_ci95_lo': round(float(ci_adni1[0]), 4),
            'adni1_ci95_hi': round(float(ci_adni1[1]), 4),
            'later_gain': round(float(later_gain), 4),
            'later_ci95_lo': round(float(ci_later[0]), 4),
            'later_ci95_hi': round(float(ci_later[1]), 4),
            'diff_adni1_minus_later': round(float(diff_gain), 4),
            'diff_ci95_lo': round(float(ci_diff[0]), 4),
            'diff_ci95_hi': round(float(ci_diff[1]), 4)
        })

    df_out = pd.DataFrame(records)
    header_comment = "# This analysis was added after the E1 results (post hoc, descriptive, not in plan 09c)\n"
    with open(out_csv, 'w') as f:
        f.write(header_comment)
        df_out.to_csv(f, index=False)

    print(f"Saved {out_csv} with {len(df_out)} contrasts.")

if __name__ == '__main__':
    compute_heterogeneity()
