import re, os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
S = coh[coh.RID.isin(dev.rid.unique()) & coh.RID.isin(ok)]; S = S[S.RID.map(lambda r: os.path.exists(f'data/interim/mri_mni2mm_v2/{r}.npy'))].reset_index(drop=True)
y = S.label_primary.astype(int).values; n = len(S); idx = {r: i for i, r in enumerate(S.RID)}; grp = np.where(S.adni_phase == 'ADNI1', 'ADNI1', 'later')
print(f'MRI-usable development participants: {n}, events {y.sum()}')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(S.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); R = f29.reindex(S.RID)
c29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; X29 = R[c29].astype(float).copy()
for c in c29:
    if c.endswith('SV'): X29[c] = X29[c] / R['ST10CV'] * 1000
X29 = X29.values; ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(S.RID)
mfs = (ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1)).astype(float)
PH = pd.get_dummies(S.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values
BT = None
if os.path.exists('results/mri_v2/scan_provenance2.csv'):
    pv = pd.read_csv('results/mri_v2/scan_provenance2.csv').drop_duplicates('RID').set_index('RID').reindex(S.RID)
    if pv.root.notna().all() and (pv.root != 'NOT_FOUND').all(): BT = (pv.root.values != 'batch3_combined').astype(float)[:, None]; print('download-round indicator available')
if BT is None: print('download-round indicator NOT available; strong reference uses phase only')
T = np.stack([np.load(f'data/interim/mri_mni2mm_v2/{r}.npy')[0].astype(np.float32)[::4, ::4, ::4].ravel() for r in S.RID]); print('tensor matrix', T.shape)
def prep(X, trn, m):
    t = trn[m[trn] > 0]; med = np.nanmedian(X[t], 0); Z = np.where(np.isnan(X), med, X); Z = (Z - Z[t].mean(0)) / (Z[t].std(0) + 1e-6); Z[m == 0] = 0; return Z
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = np.array(sorted(set(g.rid.values) & set(S.RID))); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(S.RID.values, v)])
STRONG_PARTS = PH if BT is None else np.hstack([PH, BT]); SN = 'ph' if BT is None else 'ph+bt'
M = ['A', f'A+{SN}', 'A+PC10', 'A+PC30', f'A+{SN}+PC10', f'A+{SN}+PC30']; res = {m: {} for m in M}; reps = sorted(dev.repeat.unique())
for rep in reps:
    for m in M: res[m][rep] = np.full(n, np.nan)
    for val, trn in folds(rep):
        base = np.hstack([prep(Xc, trn, np.ones(n)), prep(X29, trn, mfs), mfs[:, None]])
        pca = PCA(30, svd_solver='full', random_state=0).fit(T[trn]); P = pca.transform(T); P = (P - P[trn].mean(0)) / (P[trn].std(0) + 1e-9)
        for m in M:
            X = base
            if f'+{SN}' in m: X = np.hstack([X, STRONG_PARTS])
            if 'PC10' in m: X = np.hstack([X, P[:, :10]])
            if 'PC30' in m: X = np.hstack([X, P[:, :30]])
            clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(X[trn], y[trn]); res[m][rep][val] = clf.predict_proba(X[val])[:, 1]
pooled = {m: np.nanmean([res[m][r] for r in reps], 0) for m in M}
print('\nAUROC (pooled over repeats):', {m: round(AUC(y, pooled[m]), 4) for m in M})
B = []
while len(B) < 2000:
    i = rng.integers(0, n, n)
    if 0 < y[i].sum() < n: B.append(i)
def dci(a, b, mask=None):
    ids = np.arange(n) if mask is None else np.where(mask)[0]; yy = y[ids]; d = []
    for i in B:
        j = rng.integers(0, len(ids), len(ids))
        if 0 < yy[j].sum() < len(j): d.append(AUC(yy[j], pooled[a][ids][j]) - AUC(yy[j], pooled[b][ids][j]))
    return AUC(yy, pooled[a][ids]) - AUC(yy, pooled[b][ids]), np.percentile(d, [2.5, 97.5]), np.percentile(d, [0.5, 99.5])
print('\n=== Paired increments: with image components minus without (identical folds) ===')
for ref, new in [('A', 'A+PC10'), ('A', 'A+PC30'), (f'A+{SN}', f'A+{SN}+PC10'), (f'A+{SN}', f'A+{SN}+PC30')]:
    dl, c95, c99 = dci(new, ref); per = [AUC(y, res[new][r]) - AUC(y, res[ref][r]) for r in reps]
    g1, _, _ = dci(new, ref, grp == 'ADNI1'); g2, _, _ = dci(new, ref, grp == 'later')
    crit = dl >= 0.010 and c99[0] > 0 and sum(p > 0 for p in per) >= 2 and g1 > 0 and g2 > 0
    print(f'{new:14s} - {ref:8s}: {dl:+.4f} 95% [{c95[0]:+.4f}, {c95[1]:+.4f}] 99% [{c99[0]:+.4f}, {c99[1]:+.4f}] | per-repeat {[round(p, 4) for p in per]} | ADNI1 {g1:+.4f} later {g2:+.4f} | pre-registered criterion met: {crit}')