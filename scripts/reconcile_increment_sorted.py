import re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True); y = coh.label_primary.astype(int).values
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID'); cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(coh.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
R = fsd.reindex(coh.RID); Xf = R[fcols].astype(float).copy()
for c in fcols:
    if c.endswith('SV'): Xf[c] = Xf[c] / R['ST10CV'] * 1000
Xf = Xf.values; mfs = (ms.reindex(coh.RID).freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(Xf).all(1)).astype(float)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']; idx = {r: i for i, r in enumerate(coh.RID)}
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = set(g.rid.values); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(dev.rid.unique(), list(v))])
def go(dtype, use_fs):
    P, Cs = {}, []
    for rep in sorted(dev.repeat.unique()):
        for val, trn in folds(rep):
            med = np.nanmedian(Xc[trn], 0); Zc = np.where(np.isnan(Xc), med, Xc); Zc = (Zc - Zc[trn].mean(0)) / (Zc[trn].std(0) + 1e-6)
            t = trn[mfs[trn] > 0]; mf = np.nanmedian(Xf[t], 0); Zf = np.where(np.isnan(Xf), mf, Xf); Zf = (Zf - Zf[t].mean(0)) / (Zf[t].std(0) + 1e-6); Zf[mfs == 0] = 0
            X = (np.hstack([Zc, Zf, mfs[:, None]]) if use_fs else Zc).astype(dtype)
            clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(X[trn], y[trn]); Cs.append(float(clf.C_[0]))
            for i, v in zip(val, clf.predict_proba(X[val])[:, 1]): P.setdefault(i, []).append(v)
    ids = np.array(sorted(P)); return AUC(y[ids], [np.mean(P[i]) for i in ids]), float(np.median(Cs))
print('Reference: T3 (train_cv) clinical 0.844, clinical+fs29 0.878, increment +0.033 | T4 (sensitivity) 0.840, 0.879, +0.039')
for dt in (np.float64, np.float32):
    a0, c0 = go(dt, False); a1, c1 = go(dt, True)
    print(f'{dt.__name__}: clinical {a0:.4f} (median C {c0:.3g}) | clinical+fs29 {a1:.4f} (median C {c1:.3g}) | increment {a1 - a0:+.4f}')