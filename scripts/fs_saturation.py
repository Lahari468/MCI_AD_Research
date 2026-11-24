import re, glob, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True); y = coh.label_primary.astype(int).values
idx = {r: i for i, r in enumerate(coh.RID)}; n = len(coh)
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(coh.RID)
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(coh.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); R = f29.reindex(coh.RID)
cols29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; X29 = R[cols29].astype(float).copy()
for c in cols29:
    if c.endswith('SV'): X29[c] = X29[c] / R['ST10CV'] * 1000
X29 = X29.values
if 'IMAGEUID' in f29.columns: uid = R['IMAGEUID']
else:
    print('note: IMAGEUID not in fs29 file; using freesurfer_raw_v1 selection'); uid = pd.read_csv('data/processed/features/freesurfer_raw_v1.csv').set_index('RID')['IMAGEUID'].reindex(coh.RID)
path = glob.glob('data/raw/freesurfer/UCSFFSX7_*.csv'); assert path, 'UCSFFSX7 not found under data/raw/freesurfer'
full = pd.read_csv(path[0], low_memory=False).drop_duplicates('IMAGEUID').set_index('IMAGEUID'); rowsel = full.reindex(uid.astype(float).values)
cand = [c for c in full.columns if re.fullmatch(r'ST\d+(SV|TA|CV|SA)', c) and c != 'ST10CV']; icv = rowsel['ST10CV'].astype(float).values; F = rowsel[cand].astype(float).copy()
for c in cand:
    if c.endswith(('SV', 'CV')): F[c] = F[c] / icv * 1000
    elif c.endswith('SA'): F[c] = F[c] / icv ** (2 / 3)
keep = [c for c in cand if F[c].isna().mean() < 0.10]; other = [c for c in keep if c not in set(cols29)]; Xo = F[other].values
print(f'full FreeSurfer measures kept: {len(keep)} | beyond fs29: {len(other)}')
mfs = (ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1)).astype(float)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = np.array(sorted(g.rid.values)); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(dev.rid.unique(), v)])
def prep(X, trn, m):
    t = trn[m[trn] > 0]; med = np.nanmedian(X[t], 0); Z = np.where(np.isnan(X), med, X); Z = (Z - Z[t].mean(0)) / (Z[t].std(0) + 1e-6); Z[m == 0] = 0; return Z
def build(kind, trn):
    Zc = prep(Xc, trn, np.ones(n))
    if kind == 'clin': return Zc
    B = [Zc, prep(X29, trn, mfs), mfs[:, None]]
    if kind == 'A': return np.hstack(B)
    Zo = prep(Xo, trn, mfs)
    if kind == 'B': return np.hstack(B + [Zo])
    t = trn[mfs[trn] > 0]; Zp = PCA(10, random_state=0).fit(Zo[t]).transform(Zo); Zp[mfs == 0] = 0; return np.hstack(B + [Zp])
res = {}
for kind in ['clin', 'A', 'B', 'C']:
    P = {}
    for rep in sorted(dev.repeat.unique()):
        for val, trn in folds(rep):
            X = build(kind, trn); clf = LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=5, scoring='roc_auc', max_iter=5000).fit(X[trn], y[trn])
            for i, v in zip(val, clf.predict_proba(X[val])[:, 1]): P.setdefault(i, []).append(v)
    ids = np.array(sorted(P)); res[kind] = np.array([np.mean(P[i]) for i in ids])
yy = y[ids]; Bt = []
while len(Bt) < 2000:
    i = rng.integers(0, len(yy), len(yy))
    if 0 < yy[i].sum() < len(i): Bt.append(i)
names = {'clin': 'clinical only', 'A': 'clinical + fs29 (reference)', 'B': 'A + all other FreeSurfer measures', 'C': 'A + 10 PCs of other measures'}
print(f'\ndevelopment n={len(yy)} pos={yy.sum()}')
for k in names: print(f'{names[k]:40s} AUROC {AUC(yy, res[k]):.4f}')
for k in ['B', 'C']:
    dd = [AUC(yy[i], res[k][i]) - AUC(yy[i], res['A'][i]) for i in Bt]
    print(f'{names[k]} minus reference: {AUC(yy, res[k]) - AUC(yy, res["A"]):+.4f}  95% [{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]  99% [{np.percentile(dd, 0.5):+.4f}, {np.percentile(dd, 99.5):+.4f}]')