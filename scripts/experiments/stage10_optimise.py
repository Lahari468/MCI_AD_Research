#!/usr/bin/env python
"""
Stage 10: Three-Modality Model Optimisation and Development Benchmark
Predicting 24-month MCI-to-AD Progression (label_primary)
Strictly restricted to Development Participants (N=622, locked test set N=156 untouched).
"""

import os, sys, time, json, copy, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"[{time.strftime('%X')}] Stage 10 initialising on device: {DEV}")

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

# -------------------------------------------------------------
# 3. Load & Pre-Scale Development MRI Tensors
# -------------------------------------------------------------
mri_dir = 'data/interim/mri_mni2mm_v2'
MR_scaled = {}
t0_mri = time.time()
for r in coh.RID:
    fpath = f'{mri_dir}/{r}.npy'
    if os.path.exists(fpath):
        arr = np.load(fpath) # shape: (1, 91, 109, 91), float16
        arr_t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        # Pre-scale to scale_factor=0.67 -> shape: (1, 61, 73, 61)
        arr_s = F.interpolate(arr_t, scale_factor=0.67, mode='trilinear', align_corners=False).squeeze(0).to(torch.float16).numpy()
        MR_scaled[r] = arr_s

assert not any(r in test_rids for r in MR_scaled.keys()), "CRITICAL: Locked test MRI loaded!"
m_mri = np.array([r in MR_scaled for r in coh.RID], np.float32)
M0 = np.stack([np.ones(len(coh), np.float32), m_fs, m_mri], axis=1)

print(f"Loaded and pre-scaled {len(MR_scaled)} development MRIs in {time.time()-t0_mri:.2f}s "
      f"(Presence: Clinical 1.0, FS {m_fs.mean():.3f}, MRI {m_mri.mean():.3f})")

# -------------------------------------------------------------
# 4. Neural Network Architecture Definition
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
    def __init__(self, c_in=1, d=64, w=(16, 32, 64, 96)):
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
    def __init__(self, dc=10, df=29, mode='attn', d=64, h=4, K=1, ch=1, use_mri=True, use_fs=True):
        super().__init__()
        self.mode = mode
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
        if self.mode == 'attn':
            att, w = self.attn(T, T, T, key_padding_mask=(m < 0.5), need_weights=True, average_attn_weights=True)
            T = self.ln1(T + att)
            T = self.ln2(T + self.ffn(T)) * m.unsqueeze(-1)

        z = self.fuse(torch.cat([T.flatten(1), m], dim=1))
        logit = self.head(z).squeeze(-1)
        return logit, w

# -------------------------------------------------------------
# 5. Training, Inner-Validation, and Leak-Free Helpers
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

def mri_batch(bi, m, is_train):
    sel = np.where(m[:, 2] > 0)[0]
    if len(sel) == 0:
        return None
    x_np = np.stack([MR_scaled[RIDS[bi[j]]] for j in sel]).astype(np.float32)
    x = torch.from_numpy(x_np).to(DEV)
    if is_train:
        # mild intensity jitter
        scale = torch.empty(len(sel), 1, 1, 1, 1, device=DEV).uniform_(0.9, 1.1)
        x = x * scale
    return x

def forward_batch(model, bi, Zc, Zf, m, is_train):
    m_copy = m.copy()
    if model.enc_f is None:
        m_copy[:, 1] = 0
    if model.enc_m is None:
        m_copy[:, 2] = 0
    mri = mri_batch(bi, m_copy, is_train) if model.enc_m is not None else None
    mt = torch.from_numpy(m_copy).to(DEV)
    xc_t = torch.from_numpy(Zc[bi]).to(DEV)
    xf_t = torch.from_numpy(Zf[bi]).to(DEV)
    logits, attn_w = model(xc_t, xf_t, mri, mt)
    return logits, attn_w

def apply_modality_dropout(m, mask_p):
    if mask_p > 0:
        m = m.copy()
        # Modality dropout on FS and MRI (clinical is never dropped)
        drop_mask = np.random.rand(len(m), 2) < mask_p
        m[:, 1:] *= (~drop_mask)
    return m

@torch.no_grad()
def predict_probabilities(model, rows, Zc, Zf):
    model.eval()
    out = []
    for i in range(0, len(rows), 32):
        bi = rows[i:i+32]
        m = M0[bi].copy()
        logits, _ = forward_batch(model, bi, Zc, Zf, m, is_train=False)
        probs = torch.sigmoid(logits).cpu().numpy()
        out.append(probs)
    return np.concatenate(out)

def select_threshold_youden(y_true, y_prob):
    """Find threshold that maximizes Youden's J = Sensitivity + Specificity - 1 strictly on validation data."""
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
# 6. CV Fold Generator (Repeat 0, Folds 0-4)
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
# 7. Model Training Runners
# -------------------------------------------------------------
def train_deep_fold(cfg, trn, val, fold_idx, save_ckpt_path=None):
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    
    Zc, Zf = fit_prep(trn)
    
    # Inner validation split for early stopping and threshold selection (15% stratified)
    itr, iva = train_test_split(trn, test_size=0.15, stratify=Y[trn], random_state=SEED + fold_idx)
    
    model = Fusion(
        dc=10, df=29, mode=cfg['mode'], d=cfg['dim'], h=cfg['heads'],
        K=1, ch=1, use_mri=cfg['use_mri'], use_fs=cfg['use_fs']
    ).to(DEV)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    
    pos_weight = torch.tensor((len(itr) - Y[itr].sum()) / Y[itr].sum(), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_loss = 1e9
    best_state = None
    bad_epochs = 0
    patience = cfg['patience']
    
    train_history = []
    
    for ep in range(cfg['epochs']):
        model.train()
        perm = np.random.permutation(itr)
        trn_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), cfg['bs']):
            bi = perm[i:i+cfg['bs']]
            if len(bi) < 4:
                continue
            m_dropped = apply_modality_dropout(M0[bi], cfg['mask_p'])
            logits, _ = forward_batch(model, bi, Zc, Zf, m_dropped, is_train=True)
            loss = lossf(logits, torch.from_numpy(Y[bi]).float().to(DEV))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item() * len(bi)
            n_batches += 1
        
        sched.step()
        trn_loss /= len(itr)
        
        # Inner validation evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, len(iva), 32):
                bi = iva[i:i+32]
                logits, _ = forward_batch(model, bi, Zc, Zf, M0[bi], is_train=False)
                val_loss += lossf(logits, torch.from_numpy(Y[bi]).float().to(DEV)).item() * len(bi)
            val_loss /= len(iva)
            
        train_history.append({'epoch': ep, 'train_loss': trn_loss, 'val_loss': val_loss})
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if ep >= 10 and bad_epochs >= patience:
                break
                
    model.load_state_dict(best_state)
    if save_ckpt_path:
        torch.save(best_state, save_ckpt_path)
        
    # Evaluate inner validation predictions to select leak-free threshold
    p_iva = predict_probabilities(model, iva, Zc, Zf)
    thresh = select_threshold_youden(Y[iva], p_iva)
    
    # Outer fold prediction
    p_val = predict_probabilities(model, val, Zc, Zf)
    metrics = compute_metrics(Y[val], p_val, thresh)
    
    return metrics, p_val, train_history

def train_tabular_fold(trn, val, use_fs=True):
    Zc, Zf = fit_prep(trn)
    def make_X(rows):
        if not use_fs:
            return Zc[rows]
        return np.hstack([Zc[rows], Zf[rows], m_fs[rows, None]])
    
    itr, iva = train_test_split(trn, test_size=0.15, stratify=Y[trn], random_state=SEED)
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000)
    clf.fit(make_X(itr), Y[itr])
    
    p_iva = clf.predict_proba(make_X(iva))[:, 1]
    thresh = select_threshold_youden(Y[iva], p_iva)
    
    p_val = clf.predict_proba(make_X(val))[:, 1]
    metrics = compute_metrics(Y[val], p_val, thresh)
    return metrics, p_val

# -------------------------------------------------------------
# 8. Experiment Execution Suite
# -------------------------------------------------------------
out_dir = 'results/stage10'
ckpt_dir = os.path.join(out_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

# Predefined candidate hyperparameters for Three-Modality Attention Model
HPARAM_CANDIDATES = {
    'attn_cfg0_baseline': {
        'mode': 'attn', 'lr': 1e-4, 'wd': 1e-3, 'bs': 8, 'dim': 64, 'heads': 4,
        'mask_p': 0.3, 'patience': 12, 'epochs': 60, 'use_mri': True, 'use_fs': True,
        'description': 'Baseline: lr=1e-4, wd=1e-3, mask_p=0.3, dim=64, heads=4, patience=12'
    },
    'attn_cfg1_reg': {
        'mode': 'attn', 'lr': 1e-4, 'wd': 1e-2, 'bs': 8, 'dim': 64, 'heads': 4,
        'mask_p': 0.15, 'patience': 10, 'epochs': 60, 'use_mri': True, 'use_fs': True,
        'description': 'Regularized: wd=1e-2, mask_p=0.15, patience=10'
    },
    'attn_cfg2_compact': {
        'mode': 'attn', 'lr': 1e-4, 'wd': 5e-3, 'bs': 8, 'dim': 32, 'heads': 2,
        'mask_p': 0.3, 'patience': 8, 'epochs': 60, 'use_mri': True, 'use_fs': True,
        'description': 'Compact Latent: dim=32, heads=2, wd=5e-3, patience=8'
    },
    'attn_cfg3_lr3e4': {
        'mode': 'attn', 'lr': 3e-4, 'wd': 1e-2, 'bs': 8, 'dim': 64, 'heads': 4,
        'mask_p': 0.3, 'patience': 10, 'epochs': 60, 'use_mri': True, 'use_fs': True,
        'description': 'Higher LR + Reg: lr=3e-4, wd=1e-2, patience=10'
    }
}

with open(os.path.join(out_dir, 'experiment_config.json'), 'w') as f:
    json.dump({
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEV,
        'cohort_n': len(coh),
        'cohort_pos': int(coh.y.sum()),
        'candidates': HPARAM_CANDIDATES
    }, f, indent=2)

print("\n" + "="*80)
print("STAGE 10: HYPERPARAMETER SEARCH FOR THREE-MODALITY ATTENTION MODEL")
print("="*80)

hparam_records = []
all_fold_metrics = []
oof_dict = {}
all_training_logs = []

for cname, cfg in HPARAM_CANDIDATES.items():
    print(f"\n--- Running Candidate: {cname} ({cfg['description']}) ---")
    t0_cand = time.time()
    cand_oof_probs = np.zeros(len(coh))
    cand_fold_aurocs = []
    cand_fold_praucs = []
    
    for fold, trn_idx, val_idx in get_cv_folds():
        ckpt_path = os.path.join(ckpt_dir, f"{cname}_fold{fold}.pt")
        m, p_val, hist = train_deep_fold(cfg, trn_idx, val_idx, fold, save_ckpt_path=ckpt_path)
        cand_oof_probs[val_idx] = p_val
        cand_fold_aurocs.append(m['auroc'])
        cand_fold_praucs.append(m['prauc'])
        
        m_row = {'model': cname, 'fold': fold, **m}
        all_fold_metrics.append(m_row)
        
        for h in hist:
            all_training_logs.append({'model': cname, 'fold': fold, **h})
            
        print(f"  Fold {fold}: AUROC = {m['auroc']:.4f}, PR-AUC = {m['prauc']:.4f}, "
              f"Brier = {m['brier']:.4f}, Thresh = {m['threshold']:.2f}, Sens = {m['sensitivity']:.3f}, Spec = {m['specificity']:.3f}")
        
    oof_dict[cname] = cand_oof_probs
    pooled_auc = roc_auc_score(Y, cand_oof_probs)
    pooled_prauc = average_precision_score(Y, cand_oof_probs)
    mean_auc = np.mean(cand_fold_aurocs)
    std_auc = np.std(cand_fold_aurocs)
    mean_prauc = np.mean(cand_fold_praucs)
    std_prauc = np.std(cand_fold_praucs)
    
    cand_summary = {
        'candidate': cname,
        'description': cfg['description'],
        'pooled_auroc': pooled_auc,
        'pooled_prauc': pooled_prauc,
        'fold_auroc_mean': mean_auc,
        'fold_auroc_sd': std_auc,
        'fold_prauc_mean': mean_prauc,
        'fold_prauc_sd': std_prauc,
        'runtime_sec': round(time.time() - t0_cand, 1)
    }
    hparam_records.append(cand_summary)
    print(f"Candidate {cname} Summary: Pooled AUROC = {pooled_auc:.4f}, Mean Fold AUROC = {mean_auc:.4f} (+/- {std_auc:.4f})")

# Save hyperparameter comparison table
hparam_df = pd.DataFrame(hparam_records).sort_values('pooled_auroc', ascending=False)
hparam_df.to_csv(os.path.join(out_dir, 'hyperparameter_comparison.csv'), index=False)
print("\nHyperparameter Comparison Table:")
print(hparam_df.to_string(index=False))

# Identify selected best configuration
best_cand_name = hparam_df.iloc[0]['candidate']
best_cfg = copy.deepcopy(HPARAM_CANDIDATES[best_cand_name])
print(f"\n>>> Selected Best Three-Modality Attention Configuration: {best_cand_name} <<<")

with open(os.path.join(out_dir, 'selected_model_config.json'), 'w') as f:
    json.dump({'best_candidate': best_cand_name, 'config': best_cfg}, f, indent=2)

# -------------------------------------------------------------
# 9. Required Model Comparisons
# A. Clinical-only baseline
# B. Clinical + FS29 baseline
# C. Clinical + MRI + FS29 with Concatenation
# D. Clinical + MRI + FS29 with Attention Fusion (Selected Best)
# -------------------------------------------------------------
print("\n" + "="*80)
print("STAGE 10: REQUIRED MODEL COMPARISONS ON IDENTICAL DEVELOPMENT FOLDS")
print("="*80)

# Run Concatenation Fusion with identical architecture width/lr/wd
concat_cfg = copy.deepcopy(best_cfg)
concat_cfg['mode'] = 'concat'
concat_cfg['description'] = f"Concatenation Fusion ({best_cand_name} settings)"

print("\n--- Running Model C: Clinical + MRI + FS29 (Concatenation Fusion) ---")
concat_oof_probs = np.zeros(len(coh))
for fold, trn_idx, val_idx in get_cv_folds():
    ckpt_path = os.path.join(ckpt_dir, f"concat_mri_fold{fold}.pt")
    m, p_val, hist = train_deep_fold(concat_cfg, trn_idx, val_idx, fold, save_ckpt_path=ckpt_path)
    concat_oof_probs[val_idx] = p_val
    m_row = {'model': '3mod_concatenation', 'fold': fold, **m}
    all_fold_metrics.append(m_row)
    for h in hist:
        all_training_logs.append({'model': '3mod_concatenation', 'fold': fold, **h})
    print(f"  Fold {fold}: AUROC = {m['auroc']:.4f}, PR-AUC = {m['prauc']:.4f}, Brier = {m['brier']:.4f}")
oof_dict['3mod_concatenation'] = concat_oof_probs

# Run Clinical-Only Baseline
print("\n--- Running Model A: Clinical-Only Baseline ---")
clin_oof_probs = np.zeros(len(coh))
for fold, trn_idx, val_idx in get_cv_folds():
    m, p_val = train_tabular_fold(trn_idx, val_idx, use_fs=False)
    clin_oof_probs[val_idx] = p_val
    m_row = {'model': 'clinical_only', 'fold': fold, **m}
    all_fold_metrics.append(m_row)
    print(f"  Fold {fold}: AUROC = {m['auroc']:.4f}, PR-AUC = {m['prauc']:.4f}, Brier = {m['brier']:.4f}")
oof_dict['clinical_only'] = clin_oof_probs

# Run Clinical + FS29 Baseline
print("\n--- Running Model B: Clinical + FS29 Baseline ---")
clin_fs_oof_probs = np.zeros(len(coh))
for fold, trn_idx, val_idx in get_cv_folds():
    m, p_val = train_tabular_fold(trn_idx, val_idx, use_fs=True)
    clin_fs_oof_probs[val_idx] = p_val
    m_row = {'model': 'clinical_fs29', 'fold': fold, **m}
    all_fold_metrics.append(m_row)
    print(f"  Fold {fold}: AUROC = {m['auroc']:.4f}, PR-AUC = {m['prauc']:.4f}, Brier = {m['brier']:.4f}")
oof_dict['clinical_fs29'] = clin_fs_oof_probs

# -------------------------------------------------------------
# 10. Compile Fold Metrics, OOF Predictions, and Model Comparison
# -------------------------------------------------------------
fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)

oof_df = pd.DataFrame({'RID': RIDS, 'y_true': Y, **oof_dict})
oof_df.to_csv(os.path.join(out_dir, 'oof_predictions.csv'), index=False)

pd.DataFrame(all_training_logs).to_csv(os.path.join(out_dir, 'training_log.csv'), index=False)

# Bootstrap CI generator on pooled OOF predictions (B=1000 resamples)
def bootstrap_ci(y_true, y_prob, n_boot=1000):
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

comparison_models = [
    ('clinical_only', 'A. Clinical-only Baseline (Linear)'),
    ('clinical_fs29', 'B. Clinical + FS29 Baseline (Linear)'),
    ('3mod_concatenation', 'C. Clinical + MRI + FS29 (Concatenation)'),
    ('attn_cfg0_baseline', 'D1. Clinical + MRI + FS29 (Attention Baseline)'),
    (best_cand_name, f"D2. Clinical + MRI + FS29 (Optimised Attention: {best_cand_name})")
]

comp_rows = []
for mkey, mlabel in comparison_models:
    probs = oof_dict[mkey]
    f_sub = fold_metrics_df[fold_metrics_df.model == mkey]
    
    # Threshold for pooled predictions chosen on whole OOF
    pooled_thresh = select_threshold_youden(Y, probs)
    p_met = compute_metrics(Y, probs, pooled_thresh)
    auc_ci, prauc_ci, brier_ci = bootstrap_ci(Y, probs)
    
    comp_rows.append({
        'model_id': mkey,
        'model_name': mlabel,
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
print("FINAL STAGE 10 MODEL COMPARISON SUMMARY")
print("="*80)
cols_show = ['model_name', 'pooled_auroc', 'auroc_ci95', 'pooled_prauc', 'prauc_ci95', 'fold_auroc_mean', 'fold_auroc_sd', 'balanced_accuracy', 'f1']
print(comp_df[cols_show].to_string(index=False))

print(f"\n[{time.strftime('%X')}] Stage 10 experiments complete. Artifacts saved to {out_dir}/")
