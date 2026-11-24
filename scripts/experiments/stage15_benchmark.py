#!/usr/bin/env python3
"""
Stage 15: Final Controlled Predictive Model Benchmark
Evaluates regularized machine-learning models on Clinical + FreeSurfer-29 features
under leak-free 5-fold cross-validation on the development cohort (N=622).
Strictly development only: locked test set (N=156) is never accessed or loaded.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

SEED = 42
np.random.seed(SEED)

out_dir = 'results/stage15'
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Strict Data Isolation: Exclude Locked Test Set Immediately
# -------------------------------------------------------------
splits_file = 'data/processed/splits/splits_v1.csv'
sp = pd.read_csv(splits_file)

test_rids = set(sp[sp.split == 'test'].rid.unique())
dev_rids = set(sp[sp.split != 'test'].rid.unique())
assert len(test_rids) == 156, f"Expected 156 locked test participants, found {len(test_rids)}"
assert len(dev_rids) == 622, f"Expected 622 development participants, found {len(dev_rids)}"
assert len(test_rids & dev_rids) == 0, "CRITICAL ERROR: Overlap between dev and test splits!"

coh_all = pd.read_csv('data/processed/cohort_v2b.csv')
coh = coh_all[coh_all.RID.isin(dev_rids) & coh_all.label_primary.notna()].sort_values('RID').reset_index(drop=True)
assert len(coh) == 622, f"Expected 622 development cohort participants, found {len(coh)}"
assert len(set(coh.RID) & test_rids) == 0, "CRITICAL: Locked test participant present in cohort!"

coh['y'] = coh.label_primary.astype(int)
Y = coh.y.values
RIDS = coh.RID.values
rid_to_idx = {r: i for i, r in enumerate(RIDS)}
print(f"Development cohort verified: N = {len(coh)}, Positives = {coh.y.sum()} ({coh.y.mean():.1%}), Test Overlap = 0")

# -------------------------------------------------------------
# 2. Load Clinical and FreeSurfer-29 Features
# -------------------------------------------------------------
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
C_COLS = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']
Xc = cl.reindex(coh.RID)[C_COLS].astype(float).values

fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')
fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
assert len(fcols) == 29, f"Expected 29 FreeSurfer ST* columns, found {len(fcols)}"
R = fsd.reindex(coh.RID)
Xf = R[fcols].astype(float).copy()
if 'ST10CV' in R:
    for c in fcols:
        if c.endswith('SV'):
            Xf[c] = Xf[c] / R['ST10CV'] * 1000.0
Xf = Xf.values

stat = ms.reindex(coh.RID)['freesurfer_status']
fs_ok = stat.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(Xf).all(1)
m_fs = fs_ok.astype(np.float32)

print(f"Features loaded: Clinical (10), FreeSurfer (29), Missing FS mask sum = {m_fs.sum()}/{len(m_fs)}")

# -------------------------------------------------------------
# 3. Leak-Free Preprocessing Function
# -------------------------------------------------------------
def fit_transform_features(trn_idx, val_idx):
    med_c = np.nanmedian(Xc[trn_idx], axis=0)
    Zc_trn = np.where(np.isnan(Xc[trn_idx]), med_c, Xc[trn_idx])
    Zc_val = np.where(np.isnan(Xc[val_idx]), med_c, Xc[val_idx])
    
    mean_c = Zc_trn.mean(axis=0)
    std_c = Zc_trn.std(axis=0) + 1e-6
    Zc_trn = (Zc_trn - mean_c) / std_c
    Zc_val = (Zc_val - mean_c) / std_c
    
    t_fs = trn_idx[m_fs[trn_idx] > 0]
    med_f = np.nanmedian(Xf[t_fs], axis=0)
    Zf_trn = np.where(np.isnan(Xf[trn_idx]), med_f, Xf[trn_idx])
    Zf_val = np.where(np.isnan(Xf[val_idx]), med_f, Xf[val_idx])
    
    mean_f = Zf_trn[m_fs[trn_idx] > 0].mean(axis=0)
    std_f = Zf_trn[m_fs[trn_idx] > 0].std(axis=0) + 1e-6
    Zf_trn = (Zf_trn - mean_f) / std_f
    Zf_val = (Zf_val - mean_f) / std_f
    
    X_trn = np.hstack([Zc_trn, Zf_trn, m_fs[trn_idx, None]])
    X_val = np.hstack([Zc_val, Zf_val, m_fs[val_idx, None]])
    return X_trn, X_val

# -------------------------------------------------------------
# 4. Cross-Validation Setup
# -------------------------------------------------------------
dev_sp = sp[(sp.split != 'test') & (sp.repeat == 0)]
folds = sorted(dev_sp.fold.unique())

def get_cv_folds():
    for f in folds:
        g = dev_sp[dev_sp.fold == f]
        val_rids = g.rid.values
        trn_rids = np.setdiff1d(dev_sp.rid.unique(), val_rids)
        trn_indices = np.array([rid_to_idx[r] for r in trn_rids])
        val_indices = np.array([rid_to_idx[r] for r in val_rids])
        yield int(f), trn_indices, val_indices

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
    return best_t

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'prauc': float(average_precision_score(y_true, y_prob)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'decision_threshold': float(threshold)
    }

# -------------------------------------------------------------
# 5. Hyperparameter Grids for Candidate Models
# -------------------------------------------------------------
MODELS = {
    'lr_l2': {
        'name': 'L2 Logistic Regression',
        'grid': [{'C': c, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'random_state': SEED} 
                 for c in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]],
        'cls': LogisticRegression
    },
    'lr_elasticnet': {
        'name': 'Elastic Net Logistic Regression',
        'grid': [{'C': c, 'l1_ratio': r, 'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 2500, 'random_state': SEED} 
                 for c in [0.01, 0.05, 0.1, 0.5, 1.0] for r in [0.2, 0.5, 0.8]],
        'cls': LogisticRegression
    },
    'random_forest': {
        'name': 'Random Forest',
        'grid': [{'n_estimators': n, 'max_depth': d, 'min_samples_leaf': l, 'random_state': SEED, 'n_jobs': 2}
                 for n in [100, 200] for d in [4, 6, 8, None] for l in [2, 5, 10]],
        'cls': RandomForestClassifier
    },
    'hist_gradient_boosting': {
        'name': 'HistGradientBoosting (Tree Boosting)',
        'grid': [{'learning_rate': lr, 'max_iter': mi, 'max_depth': md, 'min_samples_leaf': msl, 'random_state': SEED}
                 for lr in [0.01, 0.05, 0.1] for mi in [50, 100, 150] for md in [3, 5] for msl in [10, 20]],
        'cls': HistGradientBoostingClassifier
    },
    'svm_rbf': {
        'name': 'RBF Support Vector Machine',
        'grid': [{'C': c, 'gamma': g, 'probability': True, 'random_state': SEED}
                 for c in [0.1, 1.0, 10.0] for g in ['scale', 0.01, 0.05, 0.1]],
        'cls': SVC
    }
}

# -------------------------------------------------------------
# 6. Execute Benchmark across Folds
# -------------------------------------------------------------
all_fold_metrics = []
oof_probs = {m: np.zeros(len(coh)) for m in MODELS}
oof_preds = {m: np.zeros(len(coh)) for m in MODELS}
selected_hparams = []

print("\nStarting Stage 15 Controlled Benchmark Evaluation...")

for fold, trn_idx, val_idx in get_cv_folds():
    print(f"\n--- Fold {fold} (Train: {len(trn_idx)}, Val: {len(val_idx)}) ---")
    
    # Nested inner validation split (15% of training fold)
    itr_idx, iva_idx = train_test_split(trn_idx, test_size=0.15, stratify=Y[trn_idx], random_state=SEED + fold)
    
    # Feature transformations fit strictly on inner train for model selection
    X_itr, X_iva = fit_transform_features(itr_idx, iva_idx)
    
    # Feature transformations fit on full outer train for final fold evaluation
    X_trn, X_val = fit_transform_features(trn_idx, val_idx)
    
    for m_key, m_info in MODELS.items():
        best_auroc = -1.0
        best_params = None
        
        # Inner CV grid search
        for params in m_info['grid']:
            clf = m_info['cls'](**params)
            clf.fit(X_itr, Y[itr_idx])
            p_iva = clf.predict_proba(X_iva)[:, 1]
            auc_iva = roc_auc_score(Y[iva_idx], p_iva)
            if auc_iva > best_auroc:
                best_auroc = auc_iva
                best_params = params
                
        # Refit best parameters on full outer train
        clf_outer = m_info['cls'](**best_params)
        clf_outer.fit(X_trn, Y[trn_idx])
        
        # Predict on inner val with outer model or inner model to get leak-free decision threshold
        # Using inner validation predictions to select optimal Youden threshold
        clf_inner = m_info['cls'](**best_params)
        clf_inner.fit(X_itr, Y[itr_idx])
        p_iva_best = clf_inner.predict_proba(X_iva)[:, 1]
        thresh = select_threshold_youden(Y[iva_idx], p_iva_best)
        
        # Outer evaluation
        p_val = clf_outer.predict_proba(X_val)[:, 1]
        m_val = compute_metrics(Y[val_idx], p_val, thresh)
        
        oof_probs[m_key][val_idx] = p_val
        oof_preds[m_key][val_idx] = (p_val >= thresh).astype(int)
        
        all_fold_metrics.append({
            'model': m_key,
            'model_name': m_info['name'],
            'fold': fold,
            **m_val
        })
        
        selected_hparams.append({
            'model': m_key,
            'fold': fold,
            'best_inner_auroc': float(best_auroc),
            'threshold': float(thresh),
            'params': {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in best_params.items()}
        })
        
        print(f"  [{m_info['name']}] Outer AUROC = {m_val['auroc']:.4f}, PR-AUC = {m_val['prauc']:.4f}, Brier = {m_val['brier']:.4f}, Thresh = {thresh:.2f}")

# -------------------------------------------------------------
# 7. Add 3-Modality Compact Attention Reference
# -------------------------------------------------------------
stage10_fm = pd.read_csv('results/stage10/fold_metrics.csv')
stage10_oof = pd.read_csv('results/stage10/oof_predictions.csv')

attn_ref_fm = stage10_fm[stage10_fm.model == 'attn_cfg2_compact'].copy()
for _, r in attn_ref_fm.iterrows():
    d = r.to_dict()
    d['model'] = 'attn_cfg2_compact'
    d['model_name'] = '3-Modality Compact Attention (Reference)'
    if 'threshold' in d and 'decision_threshold' not in d:
        d['decision_threshold'] = d['threshold']
    all_fold_metrics.append(d)

oof_probs['attn_cfg2_compact'] = stage10_oof['attn_cfg2_compact'].values
# Compute OOF preds for attention using fold thresholds
attn_thresh_col = 'threshold' if 'threshold' in attn_ref_fm.columns else 'decision_threshold'
attn_thresh_map = dict(zip(attn_ref_fm['fold'], attn_ref_fm[attn_thresh_col]))
attn_preds = np.zeros(len(coh), dtype=int)
for f, trn_idx, val_idx in get_cv_folds():
    t = attn_thresh_map[f]
    attn_preds[val_idx] = (oof_probs['attn_cfg2_compact'][val_idx] >= t).astype(int)
oof_preds['attn_cfg2_compact'] = attn_preds

fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
print("\nFold metrics saved to results/stage15/fold_metrics.csv")

# Save OOF predictions
oof_df = pd.DataFrame({
    'RID': RIDS,
    'y_true': Y
})
for m in list(MODELS.keys()) + ['attn_cfg2_compact']:
    oof_df[f'prob_{m}'] = oof_probs[m]
    oof_df[f'pred_{m}'] = oof_preds[m]
oof_df.to_csv(os.path.join(out_dir, 'oof_predictions.csv'), index=False)
print("OOF predictions saved to results/stage15/oof_predictions.csv")

# Save hyperparameter selections
with open(os.path.join(out_dir, 'hyperparameter_comparison.json'), 'w') as f:
    json.dump(selected_hparams, f, indent=2)

hparam_rows = []
for h in selected_hparams:
    hparam_rows.append({
        'model': h['model'],
        'fold': h['fold'],
        'best_inner_auroc': h['best_inner_auroc'],
        'threshold': h['threshold'],
        'params_summary': json.dumps(h['params'])
    })
pd.DataFrame(hparam_rows).to_csv(os.path.join(out_dir, 'hyperparameter_comparison.csv'), index=False)

# -------------------------------------------------------------
# 8. Model Comparison and Bootstrap Confidence Intervals
# -------------------------------------------------------------
def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi

all_models = list(MODELS.keys()) + ['attn_cfg2_compact']
model_names = {
    'lr_l2': 'L2 Logistic Regression',
    'lr_elasticnet': 'Elastic Net Logistic Regression',
    'random_forest': 'Random Forest',
    'hist_gradient_boosting': 'HistGradientBoosting (Tree Boosting)',
    'svm_rbf': 'RBF Support Vector Machine',
    'attn_cfg2_compact': '3-Modality Compact Attention (Reference)'
}

comparison_rows = []
for m in all_models:
    p = oof_probs[m]
    pred = oof_preds[m]
    
    auc = roc_auc_score(Y, p)
    auc_lo, auc_hi = bootstrap_metric_ci(Y, p, roc_auc_score)
    prauc = average_precision_score(Y, p)
    prauc_lo, prauc_hi = bootstrap_metric_ci(Y, p, average_precision_score)
    brier = brier_score_loss(Y, p)
    brier_lo, brier_hi = bootstrap_metric_ci(Y, p, brier_score_loss)
    
    f_sub = fold_metrics_df[fold_metrics_df.model == m]
    tn, fp, fn, tp = confusion_matrix(Y, pred, labels=[0, 1]).ravel()
    
    comparison_rows.append({
        'model_id': m,
        'model_name': model_names[m],
        'modalities': 'Clinical + FS29' if m != 'attn_cfg2_compact' else 'Clinical + MRI + FS29',
        'n': len(Y),
        'positives': int(Y.sum()),
        'pooled_auroc': float(auc),
        'auroc_ci95': f"[{auc_lo:.4f}, {auc_hi:.4f}]",
        'pooled_prauc': float(prauc),
        'prauc_ci95': f"[{prauc_lo:.4f}, {prauc_hi:.4f}]",
        'pooled_brier': float(brier),
        'brier_ci95': f"[{brier_lo:.4f}, {brier_hi:.4f}]",
        'fold_auroc_mean': float(f_sub['auroc'].mean()),
        'fold_auroc_sd': float(f_sub['auroc'].std(ddof=1)),
        'fold_prauc_mean': float(f_sub['prauc'].mean()),
        'fold_prauc_sd': float(f_sub['prauc'].std(ddof=1)),
        'accuracy': float(accuracy_score(Y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(Y, pred)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'precision': float(precision_score(Y, pred, zero_division=0)),
        'f1': float(f1_score(Y, pred, zero_division=0)),
        'mean_decision_threshold': float(f_sub['decision_threshold'].mean())
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
print("\nModel comparison summary:")
print(comp_df[['model_name', 'pooled_auroc', 'auroc_ci95', 'pooled_prauc', 'pooled_brier', 'fold_auroc_mean', 'balanced_accuracy', 'f1']].to_string(index=False))

# -------------------------------------------------------------
# 9. Paired Bootstrap Deltas vs References
# -------------------------------------------------------------
def paired_bootstrap_deltas(y_true, prob_a, prob_b, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    delta_auc, delta_pr, delta_br = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        delta_auc.append(roc_auc_score(y_true[idx], prob_b[idx]) - roc_auc_score(y_true[idx], prob_a[idx]))
        delta_pr.append(average_precision_score(y_true[idx], prob_b[idx]) - average_precision_score(y_true[idx], prob_a[idx]))
        delta_br.append(brier_score_loss(y_true[idx], prob_b[idx]) - brier_score_loss(y_true[idx], prob_a[idx]))
    
    def stat_pack(deltas, point_delta):
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_val = 2.0 * min((np.array(deltas) <= 0).mean(), (np.array(deltas) >= 0).mean())
        p_val = min(1.0, max(1.0 / n_boot, p_val))
        return point_delta, lo, hi, p_val
    
    pt_auc = roc_auc_score(y_true, prob_b) - roc_auc_score(y_true, prob_a)
    pt_pr = average_precision_score(y_true, prob_b) - average_precision_score(y_true, prob_a)
    pt_br = brier_score_loss(y_true, prob_b) - brier_score_loss(y_true, prob_a)
    
    res_auc = stat_pack(delta_auc, pt_auc)
    res_pr = stat_pack(delta_pr, pt_pr)
    res_br = stat_pack(delta_br, pt_br)
    return res_auc, res_pr, res_br

bootstrap_delta_rows = []

# Reference 1: lr_l2
ref1 = 'lr_l2'
for cand in [m for m in all_models if m != ref1]:
    (d_auc, lo_auc, hi_auc, p_auc), (d_pr, lo_pr, hi_pr, p_pr), (d_br, lo_br, hi_br, p_br) = paired_bootstrap_deltas(
        Y, oof_probs[ref1], oof_probs[cand]
    )
    bootstrap_delta_rows.append({
        'reference_model': ref1,
        'candidate_model': cand,
        'comparison': f"{cand} vs {ref1}",
        'delta_auroc': float(d_auc),
        'auroc_ci95': f"[{lo_auc:.4f}, {hi_auc:.4f}]",
        'p_value_auroc': float(p_auc),
        'delta_prauc': float(d_pr),
        'prauc_ci95': f"[{lo_pr:.4f}, {hi_pr:.4f}]",
        'p_value_prauc': float(p_pr),
        'delta_brier': float(d_br),
        'brier_ci95': f"[{lo_br:.4f}, {hi_br:.4f}]",
        'p_value_brier': float(p_br),
        'statistically_significant': bool(lo_auc > 0 or hi_auc < 0)
    })

# Reference 2: attn_cfg2_compact
ref2 = 'attn_cfg2_compact'
for cand in [m for m in all_models if m != ref2]:
    (d_auc, lo_auc, hi_auc, p_auc), (d_pr, lo_pr, hi_pr, p_pr), (d_br, lo_br, hi_br, p_br) = paired_bootstrap_deltas(
        Y, oof_probs[ref2], oof_probs[cand]
    )
    bootstrap_delta_rows.append({
        'reference_model': ref2,
        'candidate_model': cand,
        'comparison': f"{cand} vs {ref2}",
        'delta_auroc': float(d_auc),
        'auroc_ci95': f"[{lo_auc:.4f}, {hi_auc:.4f}]",
        'p_value_auroc': float(p_auc),
        'delta_prauc': float(d_pr),
        'prauc_ci95': f"[{lo_pr:.4f}, {hi_pr:.4f}]",
        'p_value_prauc': float(p_pr),
        'delta_brier': float(d_br),
        'brier_ci95': f"[{lo_br:.4f}, {hi_br:.4f}]",
        'p_value_brier': float(p_br),
        'statistically_significant': bool(lo_auc > 0 or hi_auc < 0)
    })

paired_df = pd.DataFrame(bootstrap_delta_rows)
paired_df.to_csv(os.path.join(out_dir, 'paired_bootstrap_deltas.csv'), index=False)
print("\nPaired bootstrap deltas saved to results/stage15/paired_bootstrap_deltas.csv")

# -------------------------------------------------------------
# 10. Save Experiment Configuration
# -------------------------------------------------------------
config = {
    'stage': 15,
    'objective': 'Final Controlled Predictive Model Benchmark',
    'cohort': 'ADNI MCI Baseline Cohort (Development Subset)',
    'n_total': 622,
    'n_positives': int(Y.sum()),
    'locked_test_status': 'Strictly Excluded (N=156 holdout never touched)',
    'n_folds': 5,
    'features': {
        'clinical_count': 10,
        'freesurfer_count': 29,
        'freesurfer_missing_indicator': 1,
        'total_feature_dim': 40
    },
    'models_evaluated': list(MODELS.keys()) + ['attn_cfg2_compact'],
    'environment_notes': {
        'scikit_learn_version': '1.6.1',
        'xgboost_status': 'Not installed in environment; HistGradientBoostingClassifier evaluated as equivalent tree boosting estimator'
    },
    'seed': SEED
}

with open(os.path.join(out_dir, 'experiment_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\nStage 15 benchmark execution completed successfully!")
