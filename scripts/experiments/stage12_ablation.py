#!/usr/bin/env python
"""
Stage 12: Matched Three-Modality MRI Ablation Experiment
Comparing:
- Model A: Clinical + FS29 Compact Attention Network (MRI excluded)
- Model B: Clinical + MRI + FS29 Compact Attention Network (Three-modality)
Within the identical deep-learning architecture and development CV protocol (N=622).
"""

import os, sys, time, json, copy, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"[{time.strftime('%X')}] Stage 12 initialising on device: {DEV}")

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

# Load development MRI presence mask for modality representation
mri_dir = 'data/interim/mri_mni2mm_v2'
mri_files = set(os.listdir(mri_dir))
m_mri = np.array([f"{r}.npy" in mri_files for r in coh.RID], np.float32)
M0 = np.stack([np.ones(len(coh), np.float32), m_fs, m_mri], axis=1)

print(f"Feature matrices loaded: Clinical (N={len(coh)}, 10), FreeSurfer (N={len(coh)}, 29), "
      f"MRI presence mean = {m_mri.mean():.3f}")

# -------------------------------------------------------------
# 3. Model Architecture (Compact Attention Framework)
# -------------------------------------------------------------
class MLPEnc(nn.Module):
    def __init__(self, d_in, d, p=(0.3, 0.2, 0.1)):
        super().__init__()
        dims = [d_in, 128, 256, d]
        layers = []
        for i in range(3):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(p[i])
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN3D(nn.Module):
    def __init__(self, c_in=1, d=32, w=(16, 32, 64, 96)):
        super().__init__()
        layers = []
        c = c_in
        for o in w:
            layers += [
                nn.Conv3d(c, o, 3, stride=2, padding=1),
                nn.GroupNorm(4, o),
                nn.GELU(),
                nn.Conv3d(o, o, 3, padding=1),
                nn.GroupNorm(4, o),
                nn.GELU()
            ]
            c = o
        self.features = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Linear(2 * c, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        h = self.features(x)
        pooled = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1)
        return self.out(pooled)

class Fusion(nn.Module):
    def __init__(self, dc=10, df=29, d=32, h=2, ch=1, use_mri=False, use_fs=True):
        super().__init__()
        self.use_mri = use_mri
        self.enc_c = MLPEnc(dc, d)
        self.enc_f = MLPEnc(df, d) if use_fs else None
        self.enc_m = CNN3D(ch, d) if use_mri else None
        self.temb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d, d)
        )
        H = 2 * d
        self.fuse = nn.Sequential(
            nn.Linear(3 * d + 3, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(H, H // 2),
            nn.LayerNorm(H // 2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(H // 2, 1)

    def forward(self, xc, xf, mri, m):
        ec = self.enc_c(xc)
        ef = self.enc_f(xf) if self.enc_f is not None else torch.zeros_like(ec)
        em = torch.zeros_like(ec)
        if self.enc_m is not None and mri is not None:
            mask_mri_bool = (m[:, 2] > 0.5)
            if mask_mri_bool.any():
                em[mask_mri_bool] = self.enc_m(mri)

        T = (torch.stack([ec, ef, em], dim=1) + self.temb) * m.unsqueeze(-1)
        w = None
        att, w = self.attn(T, T, T, key_padding_mask=(m < 0.5), need_weights=True, average_attn_weights=True)
        T = self.ln1(T + att)
        T = self.ln2(T + self.ffn(T)) * m.unsqueeze(-1)

        z = self.fuse(torch.cat([T.flatten(1), m], dim=1))
        logit = self.head(z).squeeze(-1)
        return logit, w

# -------------------------------------------------------------
# 4. Training and Evaluation Helpers
# -------------------------------------------------------------
def fit_prep(trn):
    med = np.nanmedian(Xc[trn], axis=0)
    Zc = np.where(np.isnan(Xc), med, Xc)
    mean_c = Zc[trn].mean(axis=0)
    std_c = Zc[trn].std(axis=0) + 1e-6
    Zc = (Zc - mean_c) / std_c

    t_fs = trn[m_fs[trn] > 0]
    med_f = np.nanmedian(Xf[t_fs], axis=0)
    Zf = np.where(np.isnan(Xf), med_f, Xf)
    mean_f = Zf[t_fs].mean(axis=0)
    std_f = Zf[t_fs].std(axis=0) + 1e-6
    Zf = (Zf - mean_f) / std_f
    Zf[m_fs == 0] = 0.0

    return Zc.astype(np.float32), Zf.astype(np.float32)

def forward_batch_no_mri(model, bi, Zc, Zf, m):
    m_copy = m.copy()
    m_copy[:, 2] = 0.0  # MRI strictly inactive
    mt = torch.from_numpy(m_copy).to(DEV)
    xc_t = torch.from_numpy(Zc[bi]).to(DEV)
    xf_t = torch.from_numpy(Zf[bi]).to(DEV)
    logits, attn_w = model(xc_t, xf_t, None, mt)
    return logits, attn_w

def apply_modality_dropout(m, mask_p):
    if mask_p > 0:
        m = m.copy()
        # Modality dropout on FS (MRI is already 0 in Model A)
        drop_mask = np.random.rand(len(m)) < mask_p
        m[:, 1] *= (~drop_mask)
    return m

@torch.no_grad()
def predict_probabilities_no_mri(model, rows, Zc, Zf):
    model.eval()
    out = []
    for i in range(0, len(rows), 32):
        bi = rows[i:i+32]
        m = M0[bi].copy()
        m[:, 2] = 0.0
        logits, _ = forward_batch_no_mri(model, bi, Zc, Zf, m)
        probs = torch.sigmoid(logits).cpu().numpy()
        out.append(probs)
    return np.concatenate(out)

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
        'auroc': auroc,
        'prauc': prauc,
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'f1': f1,
        'brier': brier,
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

# -------------------------------------------------------------
# 5. Cross-Validation Folds
# -------------------------------------------------------------
dev_sp = sp[(sp.split != 'test') & (sp.repeat == 0)]
def get_cv_folds():
    for f in sorted(dev_sp.fold.unique()):
        g = dev_sp[dev_sp.fold == f]
        val_rids = g.rid.values
        trn_rids = np.setdiff1d(dev_sp.rid.unique(), val_rids)
        trn_indices = np.array([rid_to_idx[r] for r in trn_rids])
        val_indices = np.array([rid_to_idx[r] for r in val_rids])
        yield int(f), trn_indices, val_indices

# -------------------------------------------------------------
# 6. Training Model A (Clinical + FS29 Compact Attention)
# -------------------------------------------------------------
out_dir = 'results/stage12'
ckpt_dir = os.path.join(out_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

CFG_MODEL_A = {
    'model_id': 'model_a_clin_fs',
    'model_name': 'Model A: Clinical + FS29 Compact Attention (No MRI)',
    'dim': 32,
    'heads': 2,
    'lr': 1e-4,
    'wd': 5e-3,
    'bs': 8,
    'mask_p': 0.3,
    'patience': 8,
    'epochs': 60,
    'use_mri': False,
    'use_fs': True
}

with open(os.path.join(out_dir, 'experiment_config.json'), 'w') as f:
    json.dump({
        'experiment': 'Stage 12 Matched Three-Modality MRI Ablation',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEV,
        'cohort_n': len(coh),
        'cohort_pos': int(coh.y.sum()),
        'config_model_a': CFG_MODEL_A,
        'config_model_b_reference': 'results/stage10/selected_model_config.json'
    }, f, indent=2)

print("\n" + "="*80)
print("STAGE 12: TRAINING MODEL A (CLINICAL + FS29 COMPACT ATTENTION, MRI EXCLUDED)")
print("="*80)

model_a_oof_probs = np.zeros(len(coh))
model_a_fold_metrics = []
training_logs = []

t0_train = time.time()
for fold, trn_idx, val_idx in get_cv_folds():
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    
    Zc, Zf = fit_prep(trn_idx)
    itr, iva = train_test_split(trn_idx, test_size=0.15, stratify=Y[trn_idx], random_state=SEED + fold)
    
    model = Fusion(dc=10, df=29, d=32, h=2, ch=1, use_mri=False, use_fs=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG_MODEL_A['lr'], weight_decay=CFG_MODEL_A['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG_MODEL_A['epochs'])
    
    pos_weight = torch.tensor((len(itr) - Y[itr].sum()) / Y[itr].sum(), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_loss = 1e9
    best_state = None
    bad_epochs = 0
    patience = CFG_MODEL_A['patience']
    
    for ep in range(CFG_MODEL_A['epochs']):
        model.train()
        perm = np.random.permutation(itr)
        trn_loss = 0.0
        for i in range(0, len(perm), CFG_MODEL_A['bs']):
            bi = perm[i:i+CFG_MODEL_A['bs']]
            if len(bi) < 4:
                continue
            m_dropped = apply_modality_dropout(M0[bi], CFG_MODEL_A['mask_p'])
            logits, _ = forward_batch_no_mri(model, bi, Zc, Zf, m_dropped)
            loss = lossf(logits, torch.from_numpy(Y[bi]).float().to(DEV))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item() * len(bi)
        
        sched.step()
        trn_loss /= len(itr)
        
        # Inner validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, len(iva), 32):
                bi = iva[i:i+32]
                logits, _ = forward_batch_no_mri(model, bi, Zc, Zf, M0[bi])
                val_loss += lossf(logits, torch.from_numpy(Y[bi]).float().to(DEV)).item() * len(bi)
            val_loss /= len(iva)
            
        training_logs.append({'model': 'model_a_clin_fs', 'fold': fold, 'epoch': ep, 'train_loss': trn_loss, 'val_loss': val_loss})
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if ep >= 10 and bad_epochs >= patience:
                break
                
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(ckpt_dir, f"model_a_fold{fold}.pt"))
    
    # Inner-validation threshold selection
    p_iva = predict_probabilities_no_mri(model, iva, Zc, Zf)
    thresh = select_threshold_youden(Y[iva], p_iva)
    
    # Outer fold prediction
    p_val = predict_probabilities_no_mri(model, val_idx, Zc, Zf)
    model_a_oof_probs[val_idx] = p_val
    
    m_val = compute_metrics(Y[val_idx], p_val, thresh)
    model_a_fold_metrics.append({'model': 'model_a_clin_fs', 'fold': fold, **m_val})
    
    print(f"  Fold {fold}: AUROC = {m_val['auroc']:.4f}, PR-AUC = {m_val['prauc']:.4f}, "
          f"Brier = {m_val['brier']:.4f}, Thresh = {m_val['threshold']:.2f}, Sens = {m_val['sensitivity']:.3f}, Spec = {m_val['specificity']:.3f}")

print(f"Model A training complete in {time.time()-t0_train:.1f}s.")

# -------------------------------------------------------------
# 7. Load Matched Model B (Stage 10 Compact Attention)
# -------------------------------------------------------------
stage10_oof = pd.read_csv('results/stage10/oof_predictions.csv')
stage10_fm = pd.read_csv('results/stage10/fold_metrics.csv')

model_b_oof_probs = stage10_oof['attn_cfg2_compact'].values
model_b_fold_metrics = stage10_fm[stage10_fm.model == 'attn_cfg2_compact'].to_dict(orient='records')
for r in model_b_fold_metrics:
    r['model'] = 'model_b_clin_mri_fs'

all_fold_metrics = model_a_fold_metrics + model_b_fold_metrics
fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)

oof_df = pd.DataFrame({
    'RID': RIDS,
    'y_true': Y,
    'model_a_clin_fs': model_a_oof_probs,
    'model_b_clin_mri_fs': model_b_oof_probs
})
oof_df.to_csv(os.path.join(out_dir, 'oof_predictions.csv'), index=False)

pd.DataFrame(training_logs).to_csv(os.path.join(out_dir, 'training_log.csv'), index=False)

# -------------------------------------------------------------
# 8. Compute Model Comparison Table
# -------------------------------------------------------------
def bootstrap_ci(y_true, y_prob, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs, praucs, briers = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        praucs.append(average_precision_score(y_true[idx], y_prob[idx]))
        briers.append(brier_score_loss(y_true[idx], y_prob[idx]))
    return (
        np.percentile(aucs, [2.5, 97.5]),
        np.percentile(praucs, [2.5, 97.5]),
        np.percentile(briers, [2.5, 97.5])
    )

comp_models = [
    ('model_a_clin_fs', 'Model A: Clinical + FS29 Compact Attention (No MRI)', model_a_oof_probs),
    ('model_b_clin_mri_fs', 'Model B: Clinical + MRI + FS29 Compact Attention (Three-Modality)', model_b_oof_probs)
]

comp_rows = []
for mid, mname, probs in comp_models:
    f_sub = fold_metrics_df[fold_metrics_df.model == mid]
    pooled_thresh = select_threshold_youden(Y, probs)
    p_met = compute_metrics(Y, probs, pooled_thresh)
    auc_ci, prauc_ci, brier_ci = bootstrap_ci(Y, probs)
    
    comp_rows.append({
        'model_id': mid,
        'model_name': mname,
        'n': len(Y),
        'positives': int(Y.sum()),
        'pooled_auroc': p_met['auroc'],
        'auroc_ci95': f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
        'pooled_prauc': p_met['prauc'],
        'prauc_ci95': f"[{prauc_ci[0]:.3f}, {prauc_ci[1]:.3f}]",
        'pooled_brier': p_met['brier'],
        'brier_ci95': f"[{brier_ci[0]:.3f}, {brier_ci[1]:.3f}]",
        'fold_auroc_mean': f_sub['auroc'].mean(),
        'fold_auroc_sd': f_sub['auroc'].std(),
        'fold_prauc_mean': f_sub['prauc'].mean(),
        'fold_prauc_sd': f_sub['prauc'].std(),
        'accuracy': p_met['accuracy'],
        'balanced_accuracy': p_met['balanced_accuracy'],
        'sensitivity': p_met['sensitivity'],
        'specificity': p_met['specificity'],
        'precision': p_met['precision'],
        'f1': p_met['f1'],
        'decision_threshold': pooled_thresh
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)

print("\n" + "="*80)
print("STAGE 12: MODEL COMPARISON TABLE")
print("="*80)
print(comp_df[['model_name', 'pooled_auroc', 'auroc_ci95', 'pooled_prauc', 'prauc_ci95', 'pooled_brier', 'fold_auroc_mean', 'balanced_accuracy', 'f1']].to_string(index=False))

# -------------------------------------------------------------
# 9. Paired Participant-Level Bootstrap Resampling (B=2000)
# Contrast: Model B (Three-Modality) minus Model A (Two-Modality)
# -------------------------------------------------------------
rng = np.random.default_rng(SEED)
B = 2000
n = len(Y)

p_a = model_a_oof_probs
p_b = model_b_oof_probs

diff_auroc = roc_auc_score(Y, p_b) - roc_auc_score(Y, p_a)
diff_prauc = average_precision_score(Y, p_b) - average_precision_score(Y, p_a)
diff_brier = brier_score_loss(Y, p_b) - brier_score_loss(Y, p_a)

boot_da = []
boot_dp = []
boot_db = []

for _ in range(B):
    idx = rng.integers(0, n, n)
    if Y[idx].sum() == 0 or Y[idx].sum() == len(idx):
        continue
    boot_da.append(roc_auc_score(Y[idx], p_b[idx]) - roc_auc_score(Y[idx], p_a[idx]))
    boot_dp.append(average_precision_score(Y[idx], p_b[idx]) - average_precision_score(Y[idx], p_a[idx]))
    boot_db.append(brier_score_loss(Y[idx], p_b[idx]) - brier_score_loss(Y[idx], p_a[idx]))

cia = np.percentile(boot_da, [2.5, 97.5])
cip = np.percentile(boot_dp, [2.5, 97.5])
cib = np.percentile(boot_db, [2.5, 97.5])

pval_a = 2 * min(np.mean(np.array(boot_da) <= 0), np.mean(np.array(boot_da) >= 0))
pval_p = 2 * min(np.mean(np.array(boot_dp) <= 0), np.mean(np.array(boot_dp) >= 0))
pval_b = 2 * min(np.mean(np.array(boot_db) <= 0), np.mean(np.array(boot_db) >= 0))

paired_delta_df = pd.DataFrame([{
    'contrast': 'Model B (Three-Modality) minus Model A (Two-Modality)',
    'subtraction_direction': 'Three-Modality (Clinical+MRI+FS) minus Two-Modality (Clinical+FS)',
    'n_participants': n,
    'n_bootstrap': B,
    'random_seed': SEED,
    'delta_auroc': diff_auroc,
    'delta_auroc_ci95_low': cia[0],
    'delta_auroc_ci95_high': cia[1],
    'delta_auroc_pval': pval_a,
    'delta_prauc': diff_prauc,
    'delta_prauc_ci95_low': cip[0],
    'delta_prauc_ci95_high': cip[1],
    'delta_prauc_pval': pval_p,
    'delta_brier': diff_brier,
    'delta_brier_ci95_low': cib[0],
    'delta_brier_ci95_high': cib[1],
    'delta_brier_pval': pval_b,
    'significant_auroc': bool(cia[0] * cia[1] > 0),
    'significant_prauc': bool(cip[0] * cip[1] > 0),
    'significant_brier': bool(cib[0] * cib[1] > 0)
}])
paired_delta_df.to_csv(os.path.join(out_dir, 'paired_bootstrap_deltas.csv'), index=False)

print("\n" + "="*80)
print("PAIRED BOOTSTRAP DELTAS (B=2000, SEED=42)")
print("="*80)
print(f"Delta AUROC: {diff_auroc:+.4f} [{cia[0]:+.4f}, {cia[1]:+.4f}], p = {pval_a:.4f}")
print(f"Delta PRAUC: {diff_prauc:+.4f} [{cip[0]:+.4f}, {cip[1]:+.4f}], p = {pval_p:.4f}")
print(f"Delta Brier: {diff_brier:+.4f} [{cib[0]:+.4f}, {cib[1]:+.4f}], p = {pval_b:.4f}")

# -------------------------------------------------------------
# 10. Fold-by-Fold Comparison Analysis
# -------------------------------------------------------------
f_comp_rows = []
for f in range(5):
    sub_a = fold_metrics_df[(fold_metrics_df.model == 'model_a_clin_fs') & (fold_metrics_df.fold == f)].iloc[0]
    sub_b = fold_metrics_df[(fold_metrics_df.model == 'model_b_clin_mri_fs') & (fold_metrics_df.fold == f)].iloc[0]
    f_comp_rows.append({
        'fold': f,
        'model_a_auroc': sub_a['auroc'],
        'model_b_auroc': sub_b['auroc'],
        'fold_delta_auroc': sub_b['auroc'] - sub_a['auroc'],
        'model_a_prauc': sub_a['prauc'],
        'model_b_prauc': sub_b['prauc'],
        'fold_delta_prauc': sub_b['prauc'] - sub_a['prauc'],
        'model_a_brier': sub_a['brier'],
        'model_b_brier': sub_b['brier'],
        'fold_delta_brier': sub_b['brier'] - sub_a['brier']
    })

fold_comp_df = pd.DataFrame(f_comp_rows)
print("\nFold-by-Fold Comparison:")
print(fold_comp_df.to_string(index=False))

print(f"\n[{time.strftime('%X')}] Stage 12 completed successfully. Outputs saved in {out_dir}/")
