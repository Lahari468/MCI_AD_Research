#!/usr/bin/env python3
"""
Targeted MRI Contribution Improvement Experiment
Evaluating the Incremental Predictive Value of Anatomically Informed Subcortical MRI ROI Features
Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-AD Progression.

Protocol: Matched 3-Repeat x 5-Fold Cross-Validation on Verified Complete MRI Development Cohort (N=564).
Output Directory: results/mri_roi_incremental/
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# -------------------------------------------------------------
# 0. Setup and Paths
# -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

OUT_DIR = 'results/mri_roi_incremental'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print(f"[{time.strftime('%X')}] Initializing Targeted MRI ROI Incremental Experiment...")

# -------------------------------------------------------------
# 1. Cohort Isolation & Dataset Loading
# -------------------------------------------------------------
sp = pd.read_csv('data/processed/splits/splits_v1.csv')
coh = pd.read_csv('data/processed/cohort_v2b.csv')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
fs = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
roi = pd.read_csv('data/processed/features/mri_subcortical_roi_features.csv').set_index('rid')
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')

dev_rids = set(sp[sp.split != 'test']['rid'].unique())
test_rids = set(sp[sp.split == 'test']['rid'].unique())

# Strict 564 matched development participants with verified usable baseline 3D MRI
s_rids = sorted(list(dev_rids.intersection(roi.index)))
assert len(s_rids) == 564, f"Expected 564 participants, got {len(s_rids)}"
assert len(set(s_rids) & test_rids) == 0, "CRITICAL: Locked holdout participant in cohort!"

s_df = coh[coh.RID.isin(s_rids)].sort_values('RID').reset_index(drop=True)
RIDS = s_df.RID.values
y = s_df.label_primary.astype(int).values
pos_count = int(y.sum())
neg_count = len(y) - pos_count
print(f"Cohort Verified: N = {len(s_df)} (Converters = {pos_count}, Non-converters = {neg_count}, Prev = {y.mean():.1%})")

# -------------------------------------------------------------
# 2. Feature Assembly
# -------------------------------------------------------------
# Clinical 10 features
c_cols = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']
Xc = cl.reindex(s_df.RID)[c_cols].astype(float).values

# FreeSurfer 29 regional morphometry normalized by intracranial volume (ST10CV)
import re
f_cols = [c for c in fs.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
assert len(f_cols) == 29, f"Expected 29 FreeSurfer ST* columns, got {len(f_cols)}"
R = fs.reindex(s_df.RID)
Xf = R[f_cols].astype(float).copy()
for c in f_cols:
    if c.endswith('SV') and 'ST10CV' in R:
        Xf[c] = Xf[c] / R['ST10CV'] * 1000.0
Xf = Xf.values

# Primary AD-Vulnerable Subcortical Target ROI Features (16 features)
target_roi_cols = [
    'hippo_left_median', 'hippo_right_median', 'hippo_bilateral_median',
    'hippo_left_std', 'hippo_right_std',
    'amygdala_left_median', 'amygdala_right_median', 'amygdala_bilateral_median',
    'amygdala_left_std', 'amygdala_right_std',
    'latvent_left_median', 'latvent_right_median', 'latvent_bilateral_median',
    'latvent_left_std', 'latvent_right_std',
    'vent_to_hippo_diff'
]
X_roi_target = roi.reindex(s_df.RID)[target_roi_cols].astype(float).values

# Extended Control ROI Features (6 features)
ctrl_roi_cols = [
    'thalamus_left_median', 'thalamus_right_median',
    'caudate_left_median', 'caudate_right_median',
    'putamen_left_median', 'putamen_right_median'
]
X_roi_ctrl = roi.reindex(s_df.RID)[ctrl_roi_cols].astype(float).values

# Define Model Feature Sets
FEATURE_SETS = {
    'clin_only': {
        'name': 'Clinical Only (10 features)',
        'X': Xc,
        'cols': c_cols
    },
    'model_a_clin_fs': {
        'name': 'Model A: Clinical 10 + FreeSurfer FS29 (Primary Baseline)',
        'X': np.hstack([Xc, Xf]),
        'cols': c_cols + f_cols
    },
    'model_b_clin_fs_roi': {
        'name': 'Model B: Clinical 10 + FreeSurfer FS29 + MRI Subcortical ROIs (Primary Target)',
        'X': np.hstack([Xc, Xf, X_roi_target]),
        'cols': c_cols + f_cols + target_roi_cols
    },
    'model_c_clin_fs_roi_ctrl': {
        'name': 'Model C: Clinical 10 + FreeSurfer FS29 + Target + Control ROIs (Extended)',
        'X': np.hstack([Xc, Xf, X_roi_target, X_roi_ctrl]),
        'cols': c_cols + f_cols + target_roi_cols + ctrl_roi_cols
    },
    'mri_roi_only': {
        'name': 'MRI Subcortical ROIs Only (16 features)',
        'X': X_roi_target,
        'cols': target_roi_cols
    }
}

# -------------------------------------------------------------
# 3. Leak-Free Preprocessing and Evaluation Helpers
# -------------------------------------------------------------
def prep_fold_matrices(X_all, trn, val):
    # Median imputation fitted strictly on inner training fold
    med = np.nanmedian(X_all[trn], axis=0)
    X_imp = np.where(np.isnan(X_all), med, X_all)
    # Z-score standardization fitted strictly on inner training fold
    mean = np.mean(X_imp[trn], axis=0)
    std = np.std(X_imp[trn], axis=0)
    std = np.where(std < 1e-4, 1.0, std)
    X_scl = (X_imp - mean) / std
    return X_scl[trn], X_scl[val]

def select_threshold_youden(y_true, y_prob):
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    auroc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    return {
        'auroc': float(auroc),
        'prauc': float(prauc),
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'brier': float(brier),
        'threshold': float(threshold),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

# -------------------------------------------------------------
# 4. Multi-Repeat 3x5 Cross-Validation Loop
# -------------------------------------------------------------
C_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

config_out = {
    'experiment': 'Targeted MRI Subcortical ROI Incremental Experiment',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'cohort_n': len(s_df),
    'cohort_pos': pos_count,
    'cohort_neg': neg_count,
    'prevalence': float(y.mean()),
    'models_evaluated': list(FEATURE_SETS.keys()),
    'c_grid': C_GRID,
    'repeats': [0, 1, 2],
    'folds_per_repeat': 5,
    'total_runs_per_model': 15
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(config_out, f, indent=2)

oof_predictions_records = []
all_fold_metrics = []
t0_exp = time.time()

print("\n" + "="*80)
print("EXECUTING 3-REPEAT 5-FOLD CV ACROSS ALL FEATURE SETS (N=564)")
print("="*80)

for rep in [0, 1, 2]:
    print(f"\n--- REPEAT {rep} ---")
    rep_sp = sp[(sp.rid.isin(s_rids)) & (sp.repeat == rep)]
    
    for fold in sorted(rep_sp.fold.unique()):
        val_rids = rep_sp[rep_sp.fold == fold]['rid'].values
        val_idx = np.array([i for i, r in enumerate(s_df.RID) if r in val_rids])
        trn_idx = np.array([i for i, r in enumerate(s_df.RID) if r not in val_rids])
        
        fold_seed = 42 + 100 * rep + fold
        
        # Nested 15% inner validation split for leak-free tuning and threshold selection
        itr_idx, iva_idx = train_test_split(trn_idx, test_size=0.15, stratify=y[trn_idx], random_state=fold_seed)
        
        fold_probs_by_model = {}
        
        for m_key, m_info in FEATURE_SETS.items():
            X_data = m_info['X']
            
            # Step A: Fit preprocessing on inner train, evaluate on inner val to pick C
            X_itr, X_iva = prep_fold_matrices(X_data, itr_idx, iva_idx)
            
            best_c = 1.0
            best_iva_auc = -1.0
            for c in C_GRID:
                clf_tune = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=fold_seed)
                clf_tune.fit(X_itr, y[itr_idx])
                p_iva_c = clf_tune.predict_proba(X_iva)[:, 1]
                auc_c = roc_auc_score(y[iva_idx], p_iva_c)
                if auc_c > best_iva_auc:
                    best_iva_auc = auc_c
                    best_c = c
            
            # Step B: Fit with best C on inner train to pick Youden threshold on inner val
            clf_inner_best = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=fold_seed)
            clf_inner_best.fit(X_itr, y[itr_idx])
            p_iva_final = clf_inner_best.predict_proba(X_iva)[:, 1]
            t_youden = select_threshold_youden(y[iva_idx], p_iva_final)
            
            # Step C: Refit on full outer train, predict on outer val
            X_trn, X_val = prep_fold_matrices(X_data, trn_idx, val_idx)
            clf_outer = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=fold_seed)
            clf_outer.fit(X_trn, y[trn_idx])
            p_val = clf_outer.predict_proba(X_val)[:, 1]
            
            m_metrics = compute_metrics(y[val_idx], p_val, t_youden)
            m_metrics.update({
                'model': m_key,
                'model_name': m_info['name'],
                'repeat': rep,
                'fold': fold,
                'best_c': best_c,
                'inner_val_auroc': round(best_iva_auc, 4)
            })
            all_fold_metrics.append(m_metrics)
            fold_probs_by_model[m_key] = p_val
            
        # Also run Secondary Nonlinear Model: HistGradientBoosting on Model B features
        X_data_nl = FEATURE_SETS['model_b_clin_fs_roi']['X']
        X_trn_nl, X_val_nl = prep_fold_matrices(X_data_nl, trn_idx, val_idx)
        X_itr_nl, X_iva_nl = prep_fold_matrices(X_data_nl, itr_idx, iva_idx)
        
        nl_clf_tune = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, random_state=fold_seed)
        nl_clf_tune.fit(X_itr_nl, y[itr_idx])
        p_iva_nl = nl_clf_tune.predict_proba(X_iva_nl)[:, 1]
        t_nl = select_threshold_youden(y[iva_idx], p_iva_nl)
        
        nl_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, random_state=fold_seed)
        nl_clf.fit(X_trn_nl, y[trn_idx])
        p_val_nl = nl_clf.predict_proba(X_val_nl)[:, 1]
        
        nl_metrics = compute_metrics(y[val_idx], p_val_nl, t_nl)
        nl_metrics.update({
            'model': 'model_b_nonlinear_gbm',
            'model_name': 'Model B: Nonlinear GBM (Clin + FS + ROI)',
            'repeat': rep,
            'fold': fold,
            'best_c': 0.05,
            'inner_val_auroc': round(roc_auc_score(y[iva_idx], p_iva_nl), 4)
        })
        all_fold_metrics.append(nl_metrics)
        fold_probs_by_model['model_b_nonlinear_gbm'] = p_val_nl
        
        # Record OOF predictions for all participants in this fold
        for i_pos, r_idx in enumerate(val_idx):
            row_oof = {
                'RID': int(RIDS[r_idx]),
                'repeat': rep,
                'fold': fold,
                'y_true': int(y[r_idx])
            }
            for m_key in fold_probs_by_model:
                row_oof[f"prob_{m_key}"] = float(fold_probs_by_model[m_key][i_pos])
            oof_predictions_records.append(row_oof)
            
        print(f"Rep {rep} Fold {fold}: "
              f"Model A (Clin+FS) AUROC = {fold_probs_by_model['model_a_clin_fs'].mean():.4f} | "
              f"Model B (+ROI) AUROC = {roc_auc_score(y[val_idx], fold_probs_by_model['model_b_clin_fs_roi']):.4f} | "
              f"Delta = {roc_auc_score(y[val_idx], fold_probs_by_model['model_b_clin_fs_roi']) - roc_auc_score(y[val_idx], fold_probs_by_model['model_a_clin_fs']):+.4f}")

print(f"\nAll 3x5 CV runs complete in {time.time()-t0_exp:.2f}s.")

# -------------------------------------------------------------
# 5. Save Outputs and Aggregations
# -------------------------------------------------------------
oof_df = pd.DataFrame(oof_predictions_records)
oof_df.to_csv(os.path.join(OUT_DIR, 'oof_predictions.csv'), index=False)

fm_df = pd.DataFrame(all_fold_metrics)
fm_df.to_csv(os.path.join(OUT_DIR, 'fold_metrics.csv'), index=False)

# Repeat-specific summary
repeat_records = []
all_models = list(FEATURE_SETS.keys()) + ['model_b_nonlinear_gbm']

for m_key in all_models:
    prob_col = f"prob_{m_key}"
    rep_aucs = []
    rep_praucs = []
    rep_briers = []
    
    for rep in [0, 1, 2]:
        sub = oof_df[oof_df.repeat == rep]
        y_r = sub['y_true'].values
        p_r = sub[prob_col].values
        
        rep_auc = roc_auc_score(y_r, p_r)
        rep_prauc = average_precision_score(y_r, p_r)
        rep_brier = brier_score_loss(y_r, p_r)
        
        rep_aucs.append(rep_auc)
        rep_praucs.append(rep_prauc)
        rep_briers.append(rep_brier)
        
        repeat_records.append({
            'model': m_key,
            'scope': f'Repeat_{rep}',
            'auroc': round(rep_auc, 4),
            'prauc': round(rep_prauc, 4),
            'brier': round(rep_brier, 4)
        })
        
    repeat_records.append({
        'model': m_key,
        'scope': 'Mean_Across_Repeats',
        'auroc': round(float(np.mean(rep_aucs)), 4),
        'prauc': round(float(np.mean(rep_praucs)), 4),
        'brier': round(float(np.mean(rep_briers)), 4),
        'auroc_sd': round(float(np.std(rep_aucs)), 4),
        'prauc_sd': round(float(np.std(rep_praucs)), 4),
        'brier_sd': round(float(np.std(rep_briers)), 4)
    })

rep_df = pd.DataFrame(repeat_records)
rep_df.to_csv(os.path.join(OUT_DIR, 'repeat_summary.csv'), index=False)

# -------------------------------------------------------------
# 6. Participant-Level Repeat-Averaged Bootstrap Comparison
# -------------------------------------------------------------
# Average each participant's predictions across the 3 repeats (N=564)
part_oof = oof_df.groupby('RID').agg({
    'y_true': 'first',
    **{f"prob_{m}": 'mean' for m in all_models}
}).reset_index()

assert len(part_oof) == 564, f"Expected 564 participants, got {len(part_oof)}"
y_p = part_oof['y_true'].values

# Primary contrast: Model B (Clin + FS + ROI) vs Model A (Clin + FS)
pA = part_oof['prob_model_a_clin_fs'].values
pB = part_oof['prob_model_b_clin_fs_roi'].values
pC = part_oof['prob_model_c_clin_fs_roi_ctrl'].values
pClin = part_oof['prob_clin_only'].values
pMRI = part_oof['prob_mri_roi_only'].values
pNL = part_oof['prob_model_b_nonlinear_gbm'].values

# Paired Bootstrap (B=2000, seed 42)
rng = np.random.default_rng(42)
B = 2000

def run_paired_bootstrap(y_true, prob_test, prob_ref):
    n = len(y_true)
    delta_auc = []
    delta_prauc = []
    delta_brier = []
    
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) > 1:
            d_a = roc_auc_score(y_true[idx], prob_test[idx]) - roc_auc_score(y_true[idx], prob_ref[idx])
            d_p = average_precision_score(y_true[idx], prob_test[idx]) - average_precision_score(y_true[idx], prob_ref[idx])
            d_b = brier_score_loss(y_true[idx], prob_test[idx]) - brier_score_loss(y_true[idx], prob_ref[idx])
            delta_auc.append(d_a)
            delta_prauc.append(d_p)
            delta_brier.append(d_b)
            
    ci_a = np.percentile(delta_auc, [2.5, 97.5])
    ci_p = np.percentile(delta_prauc, [2.5, 97.5])
    ci_b = np.percentile(delta_brier, [2.5, 97.5])
    
    p_a = 2 * min(np.mean(np.array(delta_auc) <= 0), np.mean(np.array(delta_auc) >= 0))
    p_p = 2 * min(np.mean(np.array(delta_prauc) <= 0), np.mean(np.array(delta_prauc) >= 0))
    p_b = 2 * min(np.mean(np.array(delta_brier) <= 0), np.mean(np.array(delta_brier) >= 0))
    
    pt_d_a = roc_auc_score(y_true, prob_test) - roc_auc_score(y_true, prob_ref)
    pt_d_p = average_precision_score(y_true, prob_test) - average_precision_score(y_true, prob_ref)
    pt_d_b = brier_score_loss(y_true, prob_test) - brier_score_loss(y_true, prob_ref)
    
    return {
        'delta_auroc': round(pt_d_a, 4),
        'ci_auroc_lo': round(ci_a[0], 4),
        'ci_auroc_hi': round(ci_a[1], 4),
        'p_auroc': round(p_a, 4),
        'delta_prauc': round(pt_d_p, 4),
        'ci_prauc_lo': round(ci_p[0], 4),
        'ci_prauc_hi': round(ci_p[1], 4),
        'p_prauc': round(p_p, 4),
        'delta_brier': round(pt_d_b, 4),
        'ci_brier_lo': round(ci_b[0], 4),
        'ci_brier_hi': round(ci_b[1], 4),
        'p_brier': round(p_b, 4)
    }

# Contrasts
c_b_minus_a = run_paired_bootstrap(y_p, pB, pA)
c_c_minus_a = run_paired_bootstrap(y_p, pC, pA)
c_a_minus_clin = run_paired_bootstrap(y_p, pA, pClin)
c_nl_minus_a = run_paired_bootstrap(y_p, pNL, pA)

deltas_table = pd.DataFrame([
    {
        'contrast': 'Primary: Model B (Clin+FS+ROI) minus Model A (Clin+FS)',
        'test_model': 'Model B (Clin+FS+ROI)',
        'ref_model': 'Model A (Clin+FS)',
        'test_auroc': round(roc_auc_score(y_p, pB), 4),
        'ref_auroc': round(roc_auc_score(y_p, pA), 4),
        **c_b_minus_a
    },
    {
        'contrast': 'Secondary: Model C (Clin+FS+ROI+Control) minus Model A (Clin+FS)',
        'test_model': 'Model C (+Control ROIs)',
        'ref_model': 'Model A (Clin+FS)',
        'test_auroc': round(roc_auc_score(y_p, pC), 4),
        'ref_auroc': round(roc_auc_score(y_p, pA), 4),
        **c_c_minus_a
    },
    {
        'contrast': 'Baseline Validation: Model A (Clin+FS) minus Clinical Only',
        'test_model': 'Model A (Clin+FS)',
        'ref_model': 'Clinical Only',
        'test_auroc': round(roc_auc_score(y_p, pA), 4),
        'ref_auroc': round(roc_auc_score(y_p, pClin), 4),
        **c_a_minus_clin
    },
    {
        'contrast': 'Secondary: Nonlinear GBM (Clin+FS+ROI) minus Model A (Clin+FS)',
        'test_model': 'Nonlinear GBM (+ROI)',
        'ref_model': 'Model A (Clin+FS)',
        'test_auroc': round(roc_auc_score(y_p, pNL), 4),
        'ref_auroc': round(roc_auc_score(y_p, pA), 4),
        **c_nl_minus_a
    }
])
deltas_table.to_csv(os.path.join(OUT_DIR, 'paired_bootstrap_deltas.csv'), index=False)

print("\nPaired Bootstrap Deltas Table:")
print(deltas_table[['contrast', 'test_auroc', 'ref_auroc', 'delta_auroc', 'ci_auroc_lo', 'ci_auroc_hi', 'p_auroc']].to_string(index=False))

# -------------------------------------------------------------
# 7. Secondary Exploratory Analyses: ADNI Phase Breakdown
# -------------------------------------------------------------
part_oof_phase = part_oof.merge(s_df[['RID', 'adni_phase']], on='RID')
phase_records = []

for ph, grp in part_oof_phase.groupby('adni_phase'):
    y_ph = grp['y_true'].values
    pA_ph = grp['prob_model_a_clin_fs'].values
    pB_ph = grp['prob_model_b_clin_fs_roi'].values
    
    n_ph = len(y_ph)
    pos_ph = int(y_ph.sum())
    neg_ph = n_ph - pos_ph
    
    if pos_ph > 0 and neg_ph > 0:
        auc_A_ph = roc_auc_score(y_ph, pA_ph)
        auc_B_ph = roc_auc_score(y_ph, pB_ph)
        phase_records.append({
            'phase': ph,
            'n': n_ph,
            'positives': pos_ph,
            'negatives': neg_ph,
            'model_a_auroc': round(auc_A_ph, 4),
            'model_b_auroc': round(auc_B_ph, 4),
            'delta_b_minus_a': round(auc_B_ph - auc_A_ph, 4)
        })

phase_df = pd.DataFrame(phase_records)
phase_df.to_csv(os.path.join(OUT_DIR, 'phase_breakdown.csv'), index=False)

# -------------------------------------------------------------
# 8. Secondary Exploratory Analysis: Correlation with FreeSurfer Volumes
# -------------------------------------------------------------
# Check correlation between MRI ROI features and corresponding FreeSurfer volumes
corrs = []
# ST29SV = Left Hippocampus Volume, ST88SV = Right Hippocampus Volume
# ST13SV = Left Amygdala Volume, ST72SV = Right Amygdala Volume
# ST37SV = Left Lateral Ventricle Volume, ST96SV = Right Lateral Ventricle Volume
fs_corr_pairs = [
    ('hippo_left_median', 'ST29SV', 'Left Hippocampus'),
    ('hippo_right_median', 'ST88SV', 'Right Hippocampus'),
    ('amygdala_left_median', 'ST13SV', 'Left Amygdala'),
    ('amygdala_right_median', 'ST72SV', 'Right Amygdala'),
    ('latvent_left_median', 'ST37SV', 'Left Lateral Ventricle'),
    ('latvent_right_median', 'ST96SV', 'Right Lateral Ventricle'),
]

for roi_col, fs_col, name in fs_corr_pairs:
    if fs_col in R.columns:
        roi_v = roi.reindex(s_df.RID)[roi_col].values
        fs_v = R[fs_col].values
        valid = ~np.isnan(roi_v) & ~np.isnan(fs_v)
        r_val = np.corrcoef(roi_v[valid], fs_v[valid])[0, 1]
        corrs.append({'structure': name, 'roi_feature': roi_col, 'fs_column': fs_col, 'pearson_r': round(r_val, 4)})

corr_df = pd.DataFrame(corrs)
corr_df.to_csv(os.path.join(OUT_DIR, 'fs_mri_correlation.csv'), index=False)
print("\nCorrelation with FreeSurfer Volumes:\n", corr_df.to_string(index=False))

print(f"\n[{time.strftime('%X')}] Experiment complete. All outputs saved to: {OUT_DIR}")
