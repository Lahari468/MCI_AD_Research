import re, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR
os.makedirs('results/noncog', exist_ok=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
y = coh.label_primary.astype(int).values; RIDS = coh.RID.values; idx = {r: i for i, r in enumerate(RIDS)}
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')
fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
D = cl.reindex(coh.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4']].astype(float).values
fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
R = fsd.reindex(coh.RID); F = R[fcols].astype(float).copy()
if 'ST10CV' in R:
    for c in fcols:
        if c.endswith('SV'): F[c] = F[c] / R['ST10CV'] * 1000
F = F.values
mfs = (ms.reindex(coh.RID).freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(F).all(1)).astype(np.float32)
PH = pd.get_dummies(coh.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']
test_ids = set(sp[sp.split == 'test'].rid); assert not (set(dev.rid) & test_ids)
dev_idx = np.array([idx[r] for r in dev.rid.unique()])
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = set(g.rid.values); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in dev.rid.unique() if r not in v])
def prep(X, trn, m=None):
    t = trn if m is None else trn[m[trn] > 0]; med = np.nanmedian(X[t], 0); Z = np.where(np.isnan(X), med, X)
    Z = (Z - Z[t].mean(0)) / (Z[t].std(0) + 1e-6)
    if m is not None: Z[m == 0] = 0
    return Z
SETS = {'demo(age,sex,edu,APOE4)': 'D', 'demo+phase': 'DP', 'fs29 only': 'F', 'demo+fs29': 'DF', 'demo+fs29+phase': 'DFP'}
reps = sorted(dev.repeat.unique()); OOF = {}; rows = []
for name, code in SETS.items():
    pr = np.full((len(reps), len(y)), np.nan)
    for ri, rep in enumerate(reps):
        for val, trn in folds(rep):
            assert not (set(RIDS[val]) & test_ids) and not (set(RIDS[trn]) & test_ids)
            B_ = []
            if 'D' in code: B_.append(prep(D, trn))
            if 'F' in code: B_ += [prep(F, trn, mfs), mfs[:, None]]
            if 'P' in code: B_.append(PH)
            X = np.hstack(B_)
            pr[ri, val] = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
    p = np.nanmean(pr, 0); OOF[name] = p
    ra = [AUC(y[dev_idx], pr[r][dev_idx]) for r in range(len(reps))]; yy, pp = y[dev_idx], p[dev_idx]; ph = coh.adni_phase.values[dev_idx]
    g = lambda m: round(AUC(yy[m], pp[m]), 3) if 0 < yy[m].sum() < m.sum() else np.nan
    rows.append(dict(model=name, n=len(yy), pos=int(yy.sum()), auroc_rep_mean=round(np.mean(ra), 3), rep_sd=round(np.std(ra), 3), auroc_pooled=round(AUC(yy, pp), 3),
                     prauc=round(AP(yy, pp), 3), brier=round(BR(yy, pp), 3), ADNI1=g(ph == 'ADNI1'), GO_ADNI2=g(np.isin(ph, ['ADNIGO', 'ADNI2'])), ADNI3=g(ph == 'ADNI3')))
S = pd.DataFrame(rows); S.to_csv('results/noncog/summary.csv', index=False); print(S.to_string(index=False))
pd.DataFrame({'rid': RIDS[dev_idx], **{k: v[dev_idx] for k, v in OOF.items()}}).to_csv('results/noncog/oof.csv', index=False)
rng = np.random.default_rng(0); yy = y[dev_idx]
Bt = [i for i in (rng.integers(0, len(yy), len(yy)) for _ in range(2000)) if 0 < yy[i].sum() < len(yy)]
for a, b in [('demo+fs29', 'demo(age,sex,edu,APOE4)'), ('demo+fs29+phase', 'demo+phase')]:
    pa, pb = OOF[a][dev_idx], OOF[b][dev_idx]; dl = [AUC(yy[i], pa[i]) - AUC(yy[i], pb[i]) for i in Bt]
    print(f'FS29 increment without cognitive scores [{a} - {b}]: dAUROC {AUC(yy, pa) - AUC(yy, pb):+.3f} {np.percentile(dl, [2.5, 97.5]).round(3)}')