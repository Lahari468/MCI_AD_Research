import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].set_index('RID')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid')
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')
fs = pd.read_csv('data/processed/features/freesurfer_raw_v1.csv').set_index('RID')
tst = set(sp[sp.split == 'test'].index)
rng = np.random.default_rng(0)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

print('=== A. FreeSurfer status vs flags (eligible cohort) ===')
e, f = ms.reindex(coh.index), fs.reindex(coh.index)
print(pd.crosstab(e.freesurfer_status, [f.fs_flag.astype(str), f.fs_screen_pass.astype(str)]))
print(coh.label_primary.groupby(e.freesurfer_status).agg(['size', 'mean']).round(3))

print('\n=== B. Dev vs test covariates (SMD; |SMD|>0.2 notable) ===')
X = cl.reindex(coh.index)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'FAQTOTAL', 'TOTAL13']].copy()
X['ADNI1'] = (coh.adni_phase == 'ADNI1').astype(float)
ist = pd.Series(X.index.isin(list(tst)), index=X.index)
def smd(a, b):
    a, b = a.dropna(), b.dropna(); s = np.sqrt((a.var() + b.var()) / 2); return (a.mean() - b.mean()) / s if s > 0 else np.nan
rows = []
for nm, m in [('all', coh.label_primary >= 0), ('positives', coh.label_primary == 1), ('negatives', coh.label_primary == 0)]:
    for c in X.columns:
        a, b = X.loc[ist & m, c], X.loc[~ist & m, c]; rows.append((nm, c, round(a.mean(), 2), round(b.mean(), 2), round(smd(a, b), 2)))
print(pd.DataFrame(rows, columns=['group', 'var', 'test_mean', 'dev_mean', 'SMD']).to_string(index=False))

print('\n=== C. Is test AUROC plausible under dev performance? ===')
d = pd.read_csv('results/deep/lr_full_fs29/oof.csv'); d = d[d.cond == 'full'].groupby('rid').prob.mean()
y = coh.loc[d.index, 'label_primary'].astype(int)
pos, neg = d[y == 1].values, d[y == 0].values
yy = np.r_[np.ones(36), np.zeros(120)]
a = np.array([AUC(yy, np.r_[rng.choice(pos, 36), rng.choice(neg, 120)]) for _ in range(5000)])
print(f'dev-resampled AUROC (36 pos / 120 neg): mean {a.mean():.3f} sd {a.std():.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]  P(>=0.930)={np.mean(a >= 0.930):.3f}')

print('\n=== D. Primary model on locked test: threshold metrics and calibration slope with CIs ===')
cal = LogisticRegression(C=1e6).fit(lg(d.values)[:, None], y.values)
t = pd.read_csv('results/deep/final_lr_full_fs29/oof.csv'); t = t[t.cond == 'full'].groupby('rid').prob.mean()
yt = coh.loc[t.index, 'label_primary'].astype(int).values; r = cal.predict_proba(lg(t.values)[:, None])[:, 1]
B = [i for i in (rng.integers(0, len(yt), len(yt)) for _ in range(2000)) if 0 < yt[i].sum() < len(yt)]
def mt(yy_, rr, th):
    pr = rr >= th; tp = (pr & (yy_ == 1)).sum(); fp = (pr & (yy_ == 0)).sum(); fn = (~pr & (yy_ == 1)).sum(); tn = (~pr & (yy_ == 0)).sum()
    return [tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), tn / max(tn + fn, 1)]
for th in (0.2, 0.3):
    pt = mt(yt, r, th); bs = np.array([mt(yt[i], r[i], th) for i in B])
    print(f'thr {th}: ' + '  '.join(f'{n}={v:.3f} [{lo:.3f},{hi:.3f}]' for n, v, lo, hi in zip(['sens', 'spec', 'ppv', 'npv'], pt, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0))))
sl = lambda yy_, pp: LogisticRegression(C=1e6).fit(lg(pp)[:, None], yy_).coef_[0, 0]
bsl = [sl(yt[i], t.values[i]) for i in B[:500]]
print(f'raw calibration slope on test: {sl(yt, t.values):.2f} [{np.percentile(bsl, 2.5):.2f}, {np.percentile(bsl, 97.5):.2f}]')

print('\n=== E. Negatives without a >=730-day visit ===')
neg_c = coh[coh.label_primary == 0]; print('primary negatives:', len(neg_c), '| without >=730d visit:', int(neg_c.label_sens_ge730.isna().sum()))