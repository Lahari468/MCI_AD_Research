#!/usr/bin/env python3
"""
E1: Anatomical Specificity Experiment
As specified in reports/audit/09c_experiment_plan.md
"""

import os, sys, time, json, warnings, hashlib, datetime, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
t_start = time.time()
print(f"[{time.strftime('%X')}] Starting E1 Anatomical Specificity Experiment...")

# -------------------------------------------------------------
# 0. Load Data & Assert Protection of Locked Test Set
# -------------------------------------------------------------
mf = pd.read_csv('data/processed/mri/manifests/mri_manifest_713.csv')
sp = pd.read_csv('data/processed/splits/splits_v1.csv')
coh = pd.read_csv('data/processed/cohort_v2b.csv')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
roi_ctrl = pd.read_csv('results/audit/qc/roi_features_704.csv').set_index('rid')
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')

dev_rids = set(sp[sp.split != 'test'].rid.unique())
test_rids = set(sp[sp.split == 'test'].rid.unique())
mri_usable_rids = set(mf[mf.mri_usable].rid.unique())
s_rids = sorted(list(dev_rids & mri_usable_rids))

S = coh[coh.RID.isin(s_rids)].drop_duplicates('RID').sort_values('RID').reset_index(drop=True)
assert len(S) == 564, f"Expected 564 development participants, got {len(S)}"
assert not (set(S.RID) & test_rids), "CRITICAL: Locked test set participant present in development cohort!"
print(f"Cohort verified: N = {len(S)} participants, events = {int(S.label_primary.sum())}, test overlap = 0.")

rids = S.RID.values
y = S.label_primary.astype(int).values
adni_phase = S.adni_phase.values
is_adni1 = (adni_phase == 'ADNI1')

# -------------------------------------------------------------
# 1. Feature Matrices
# -------------------------------------------------------------
# C: 10 clinical features
Xc = cl.reindex(rids)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values

# ROI18 and CTRL18
rc_df = roi_ctrl.reindex(rids)
roi_cols = [c for c in rc_df.columns if any(k in c for k in ['hippo', 'amy', 'latvent']) and c != 'image_id']
ctrl_cols = [c for c in rc_df.columns if any(k in c for k in ['thal', 'caud', 'put']) and c != 'image_id']
assert len(roi_cols) == 18, f"Expected 18 ROI cols, got {len(roi_cols)}"
assert len(ctrl_cols) == 18, f"Expected 18 CTRL cols, got {len(ctrl_cols)}"
X_roi = rc_df[roi_cols].astype(float).values
X_ctrl = rc_df[ctrl_cols].astype(float).values

# F: C + fs29 (volumes divided by ICV * 1000) + FreeSurfer-observed indicator
c29 = [c for c in f29.columns if any(c.endswith(s) for s in ['SV', 'TA', 'CV']) and c != 'FSVER']
R = f29.reindex(rids)
X29 = R[c29].astype(float).copy()
for c in c29:
    if c.endswith('SV'):
        X29[c] = X29[c] / R['ST10CV'] * 1000.0
ms_s = ms.reindex(rids)
mfs = (ms_s.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29.values).all(1)).astype(float)
X29_vals = X29.values
X29_vals[mfs == 0] = np.nan
X_f = np.hstack([Xc, X29_vals, mfs[:, None]])

# ph: 4 phase indicators
PH = pd.get_dummies(S.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values

# prot: one-hot of manufacturer x field_strength_normalized
mf_s = mf.drop_duplicates('rid').set_index('rid').reindex(rids).reset_index()
prot_series = mf_s['scanner'].astype(str) + '_' + mf_s['field_strength_normalized'].astype(str)
PROT = pd.get_dummies(prot_series).reindex(columns=['GE_1.5', 'GE_3.0', 'PHILIPS_1.5', 'PHILIPS_3.0', 'SIEMENS_1.5', 'SIEMENS_3.0'], fill_value=0).astype(float).values

B = {
    'c': Xc,
    'roi': X_roi,
    'ctrl': X_ctrl,
    'f': X_f,
    'ph': PH,
    'prot': PROT
}

MODELS = {
    'C': ['c'],
    'C+ROI18': ['c', 'roi'],
    'C+CTRL18': ['c', 'ctrl'],
    'C+ph': ['c', 'ph'],
    'C+ph+ROI18': ['c', 'ph', 'roi'],
    'C+ph+CTRL18': ['c', 'ph', 'ctrl'],
    'C+prot': ['c', 'prot'],
    'C+prot+ROI18': ['c', 'prot', 'roi'],
    'C+prot+CTRL18': ['c', 'prot', 'ctrl'],
    'F': ['f'],
    'F+ROI18': ['f', 'roi']
}

# -------------------------------------------------------------
# 2. Cross-Validation Folds Generation
# -------------------------------------------------------------
os.makedirs('results/experiments', exist_ok=True)
strata = S['label_primary'].astype(int).astype(str) + '_' + S['adni_phase'].astype(str)

fold_rows = []
folds_by_rep = {}
for rep in range(10):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + rep)
    folds_by_rep[rep] = []
    for fold, (trn, val) in enumerate(skf.split(S, strata)):
        folds_by_rep[rep].append((trn, val))
        for v in val:
            fold_rows.append({
                'repeat': rep,
                'fold': fold,
                'rid': rids[v],
                'split': 'val'
            })

folds_df = pd.DataFrame(fold_rows)
folds_df.to_csv('results/experiments/e1_folds.csv', index=False)
fold_file_hash = hashlib.sha256(open('results/experiments/e1_folds.csv', 'rb').read()).hexdigest()
print(f"Generated 10 repeats x 5 folds saved to results/experiments/e1_folds.csv (SHA-256: {fold_file_hash[:12]}...).")

# -------------------------------------------------------------
# 3. Model Pipeline Construction & Fitting
# -------------------------------------------------------------
param_grid = {'lr__C': np.logspace(-3, 1, 7)}
cv_inner = StratifiedKFold(5, shuffle=True, random_state=0)

def build_pipeline(parts):
    arrs = [B[p] for p in parts]
    X = np.hstack(arrs)
    steps = []
    s = 0
    for p, a in zip(parts, arrs):
        cols = list(range(s, s + a.shape[1]))
        s += a.shape[1]
        if p in ('ph', 'prot'):
            t = Pipeline([('s', StandardScaler())])
        else:
            t = Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())])
        steps.append((p, t, cols))
    pipe = Pipeline([
        ('pre', ColumnTransformer(steps)),
        ('lr', LogisticRegression(max_iter=3000, solver='lbfgs'))
    ])
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv_inner, n_jobs=1)
    return X, gs

print(f"[{time.strftime('%X')}] Fitting 11 models across 10 repeats x 5 folds...")
oof_preds = {m: np.zeros((10, len(S)), dtype=np.float32) for m in MODELS}

for m_name, parts in MODELS.items():
    t_m = time.time()
    X, gs = build_pipeline(parts)
    for rep in range(10):
        for trn, val in folds_by_rep[rep]:
            gs.fit(X[trn], y[trn])
            oof_preds[m_name][rep, val] = gs.predict_proba(X[val])[:, 1]
    print(f"  {m_name:14s}: fitted in {time.time()-t_m:.2f}s | pooled AUROC = {AUC(y, oof_preds[m_name].mean(0)):.4f}")

# Save out-of-fold probabilities
oof_records = []
for m_name in MODELS:
    for rep in range(10):
        for i in range(len(S)):
            oof_records.append({
                'model': m_name,
                'repeat': rep,
                'rid': rids[i],
                'prob': round(float(oof_preds[m_name][rep, i]), 6)
            })

oof_df = pd.DataFrame(oof_records)
oof_df.to_csv('results/experiments/e1_oof.csv', index=False)
print(f"Saved out-of-fold predictions to results/experiments/e1_oof.csv ({len(oof_df)} rows).")

# -------------------------------------------------------------
# 4. Model Evaluation & Bootstrap Inference on 11 Estimands
# -------------------------------------------------------------
print(f"[{time.strftime('%X')}] Computing performance metrics and paired bootstrap CIs...")
pooled_preds = {m: oof_preds[m].mean(0) for m in MODELS}

model_metrics = []
for m_name in MODELS:
    p_pool = pooled_preds[m_name]
    rep_aucs = [AUC(y, oof_preds[m_name][r]) for r in range(10)]
    auc_pool = AUC(y, p_pool)
    auc_adni1 = AUC(y[is_adni1], p_pool[is_adni1])
    auc_later = AUC(y[~is_adni1], p_pool[~is_adni1])
    model_metrics.append({
        'model': m_name,
        'pooled_auroc': round(float(auc_pool), 4),
        'repeat_mean_auroc': round(float(np.mean(rep_aucs)), 4),
        'repeat_sd_auroc': round(float(np.std(rep_aucs)), 4),
        'adni1_auroc': round(float(auc_adni1), 4),
        'later_auroc': round(float(auc_later), 4)
    })

model_summary_df = pd.DataFrame(model_metrics)
model_summary_df.to_csv('results/experiments/e1_models_summary.csv', index=False)

# 11 Non-permutation estimands
PAIRED_ESTIMANDS = [
    # Specificity contrasts (3)
    ('C+ROI18 - C+CTRL18', 'C+ROI18', 'C+CTRL18', 'specificity'),
    ('C+ph+ROI18 - C+ph+CTRL18', 'C+ph+ROI18', 'C+ph+CTRL18', 'specificity'),
    ('C+prot+ROI18 - C+prot+CTRL18', 'C+prot+ROI18', 'C+prot+CTRL18', 'specificity'),
    # Gain contrasts (6)
    ('C+ROI18 - C', 'C+ROI18', 'C', 'gain'),
    ('C+CTRL18 - C', 'C+CTRL18', 'C', 'gain'),
    ('C+ph+ROI18 - C+ph', 'C+ph+ROI18', 'C+ph', 'gain'),
    ('C+ph+CTRL18 - C+ph', 'C+ph+CTRL18', 'C+ph', 'gain'),
    ('C+prot+ROI18 - C+prot', 'C+prot+ROI18', 'C+prot', 'gain'),
    ('C+prot+CTRL18 - C+prot', 'C+prot+CTRL18', 'C+prot', 'gain'),
    # Benchmark contrasts (2)
    ('F - C', 'F', 'C', 'benchmark'),
    ('F+ROI18 - F', 'F+ROI18', 'F', 'benchmark')
]

rng = np.random.default_rng(42)
N_BOOT = 2000
boot_indices = [rng.integers(0, len(S), len(S)) for _ in range(N_BOOT)]

def paired_bootstrap_ci(pa, pb, y_arr, idx_list):
    diffs = []
    for j in idx_list:
        yj = y_arr[j]
        if 0 < yj.sum() < len(j):
            diffs.append(AUC(yj, pa[j]) - AUC(yj, pb[j]))
    return np.percentile(diffs, [2.5, 97.5])

estimand_rows = []
for label, m_a, m_b, cat in PAIRED_ESTIMANDS:
    pa = pooled_preds[m_a]
    pb = pooled_preds[m_b]
    d_pool = float(AUC(y, pa) - AUC(y, pb))
    ci95 = paired_bootstrap_ci(pa, pb, y, boot_indices)
    
    # Per-repeat differences
    rep_diffs = [float(AUC(y, oof_preds[m_a][r]) - AUC(y, oof_preds[m_b][r])) for r in range(10)]
    
    # Subgroup differences
    d_adni1 = float(AUC(y[is_adni1], pa[is_adni1]) - AUC(y[is_adni1], pb[is_adni1]))
    d_later = float(AUC(y[~is_adni1], pa[~is_adni1]) - AUC(y[~is_adni1], pb[~is_adni1]))
    
    estimand_rows.append({
        'contrast': label,
        'category': cat,
        'model_a': m_a,
        'model_b': m_b,
        'pooled_delta_auroc': round(d_pool, 4),
        'ci95_lower': round(float(ci95[0]), 4),
        'ci95_upper': round(float(ci95[1]), 4),
        'repeat_mean_delta': round(float(np.mean(rep_diffs)), 4),
        'repeat_sd_delta': round(float(np.std(rep_diffs)), 4),
        'adni1_delta': round(d_adni1, 4),
        'later_delta': round(d_later, 4)
    })

estimand_df = pd.DataFrame(estimand_rows)
estimand_df.to_csv('results/experiments/e1_estimands_summary.csv', index=False)
print("Computed 11 non-permutation estimands and saved to results/experiments/e1_estimands_summary.csv.")

# -------------------------------------------------------------
# 5. Permutation Null Distribution (B = 500)
# -------------------------------------------------------------
B_PERM = 500
print(f"[{time.strftime('%X')}] Starting Permutation Tests (B = {B_PERM}, 4 workers)...")

# Permutation strata: phase x protocol, pooling small strata (n < 10)
perm_strata = S['adni_phase'].astype(str) + '__' + prot_series
vc = perm_strata.value_counts()
small_idx = vc[vc < 10].index
perm_strata_clean = perm_strata.replace(small_idx, 'POOLED_SMALL').values
stratum_groups = [np.where(perm_strata_clean == st)[0] for st in np.unique(perm_strata_clean)]

def run_single_permutation(perm_seed):
    rng_p = np.random.default_rng(perm_seed)
    perm_idx = np.arange(len(S))
    for grp_indices in stratum_groups:
        perm_idx[grp_indices] = rng_p.permutation(grp_indices)
    
    X_roi_shuff = X_roi[perm_idx]
    
    # 1. Refit C+ROI18
    X_c_roi = np.hstack([Xc, X_roi_shuff])
    steps1 = [
        ('c', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(10))),
        ('roi', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(10, 28)))
    ]
    pipe1 = Pipeline([('pre', ColumnTransformer(steps1)), ('lr', LogisticRegression(max_iter=3000, solver='lbfgs'))])
    gs1 = GridSearchCV(pipe1, param_grid, scoring='roc_auc', cv=cv_inner, n_jobs=1)
    
    # 2. Refit C+prot+ROI18
    X_prot_roi = np.hstack([Xc, PROT, X_roi_shuff])
    steps2 = [
        ('c', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(10))),
        ('prot', Pipeline([('s', StandardScaler())]), list(range(10, 16))),
        ('roi', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(16, 34)))
    ]
    pipe2 = Pipeline([('pre', ColumnTransformer(steps2)), ('lr', LogisticRegression(max_iter=3000, solver='lbfgs'))])
    gs2 = GridSearchCV(pipe2, param_grid, scoring='roc_auc', cv=cv_inner, n_jobs=1)
    
    oof1 = np.zeros((10, len(S)), dtype=np.float32)
    oof2 = np.zeros((10, len(S)), dtype=np.float32)
    
    for rep in range(10):
        for trn, val in folds_by_rep[rep]:
            gs1.fit(X_c_roi[trn], y[trn])
            oof1[rep, val] = gs1.predict_proba(X_c_roi[val])[:, 1]
            gs2.fit(X_prot_roi[trn], y[trn])
            oof2[rep, val] = gs2.predict_proba(X_prot_roi[val])[:, 1]
            
    auc_perm_c_roi = AUC(y, oof1.mean(0))
    auc_perm_prot_roi = AUC(y, oof2.mean(0))
    
    # Gains under permutation
    null_gain_c = auc_perm_c_roi - AUC(y, pooled_preds['C'])
    null_gain_prot = auc_perm_prot_roi - AUC(y, pooled_preds['C+prot'])
    return null_gain_c, null_gain_prot

t_perm_start = time.time()
perm_seeds = [2000 + b for b in range(B_PERM)]
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting {B_PERM} permutations on 4 workers...", flush=True)

ctx = mp.get_context('fork')
results = []
with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
    futures = [executor.submit(run_single_permutation, s) for s in perm_seeds]
    for idx, fut in enumerate(as_completed(futures), 1):
        results.append(fut.result())
        if idx % 25 == 0 or idx == B_PERM:
            elapsed = time.time() - t_perm_start
            rate = elapsed / idx
            eta = (B_PERM - idx) * rate
            print(f"  Permutation {idx}/{B_PERM} done ({elapsed/60:.1f} min elapsed, ETA {eta/60:.1f} min)...", flush=True)

null_gains_c = np.array([r[0] for r in results])
null_gains_prot = np.array([r[1] for r in results])
sec_perms = time.time() - t_perm_start
print(f"Permutations complete ({B_PERM} realizations in {sec_perms/60:.2f} min).", flush=True)

np.save('results/experiments/e1_permutation_null_c_roi.npy', null_gains_c)
np.save('results/experiments/e1_permutation_null_prot_roi.npy', null_gains_prot)

obs_gain_c = float(AUC(y, pooled_preds['C+ROI18']) - AUC(y, pooled_preds['C']))
obs_gain_prot = float(AUC(y, pooled_preds['C+prot+ROI18']) - AUC(y, pooled_preds['C+prot']))

p_val_c = float((1 + np.sum(null_gains_c >= obs_gain_c)) / (B_PERM + 1))
p_val_prot = float((1 + np.sum(null_gains_prot >= obs_gain_prot)) / (B_PERM + 1))

perm_summary = [
    {
        'estimand': 'gain: C+ROI18 - C',
        'observed_gain': round(obs_gain_c, 4),
        'B': B_PERM,
        'null_mean': round(float(null_gains_c.mean()), 4),
        'null_sd': round(float(null_gains_c.std()), 4),
        'null_p95': round(float(np.percentile(null_gains_c, 95)), 4),
        'permutation_p_value': round(p_val_c, 4)
    },
    {
        'estimand': 'gain: C+prot+ROI18 - C+prot',
        'observed_gain': round(obs_gain_prot, 4),
        'B': B_PERM,
        'null_mean': round(float(null_gains_prot.mean()), 4),
        'null_sd': round(float(null_gains_prot.std()), 4),
        'null_p95': round(float(np.percentile(null_gains_prot, 95)), 4),
        'permutation_p_value': round(p_val_prot, 4)
    }
]

perm_summary_df = pd.DataFrame(perm_summary)
perm_summary_df.to_csv('results/experiments/e1_permutation_summary.csv', index=False)
print("Saved results/experiments/e1_permutation_summary.csv.")
print(f"E1 finished in {time.time()-t_start:.2f}s total.")
