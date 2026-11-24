import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
coh = pd.read_csv('data/processed/cohort_v2b.csv')
sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
test = set(sp[sp.split == 'test'].index)
pool = coh[coh.label_primary.notna() | coh.exclusion_reason.eq('INSUFFICIENT_FOLLOWUP')].copy()
pool['obs'] = pool.label_primary.notna().astype(int)
print('pool (eligible + insufficient follow-up):', len(pool), '| labelled:', pool.obs.sum())
pool = pool[~pool.RID.isin(test)]; print('after removing locked-test participants:', len(pool))
print(pool.groupby('adni_phase').obs.agg(['size', 'mean']).round(3))
n4 = int((pool.adni_phase == 'ADNI4').sum()); pool = pool[pool.adni_phase != 'ADNI4'].reset_index(drop=True)
print(f'ADNI4 removed (no eligible participants -> no positivity): {n4} | analysed pool: {len(pool)}')
cols = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'FAQTOTAL', 'TOTAL13']
X = cl.reindex(pool.RID)[cols].astype(float).reset_index(drop=True)
for ph in ['ADNI1', 'ADNIGO', 'ADNI2']: X['ph_' + ph] = (pool.adni_phase.values == ph).astype(float)
Xf = X.fillna(X.median()); Z = (Xf - Xf.mean()) / (Xf.std() + 1e-9)
m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(Z, pool.obs.values)
ps = m.predict_proba(Z)[:, 1]
print(f'weight model AUROC (in-sample): {AUC(pool.obs, ps):.3f} | P(labelled) range {ps.min():.3f}-{ps.max():.3f}')
w = pd.Series(np.where(pool.obs.values == 1, pool.obs.mean() / ps, 0.0)); obs = pool.obs.values == 1
w[obs] = w[obs].clip(upper=np.percentile(w[obs], 99)); we = w[obs]
print(f'weights (labelled): mean {we.mean():.2f} max {we.max():.2f} | effective n {we.sum() ** 2 / (we ** 2).sum():.0f} of {len(we)}')
rows = []
for c in X.columns:
    full, e = Xf[c], Xf[c][obs]; sd = full.std() + 1e-9; mw = np.average(e, weights=we)
    rows.append(dict(var=c, pool_mean=full.mean(), eligible_mean=e.mean(), eligible_weighted=mw, SMD_unweighted=(e.mean() - full.mean()) / sd, SMD_weighted=(mw - full.mean()) / sd))
print('\nCovariate balance vs full pool:'); print(pd.DataFrame(rows).round(2).to_string(index=False))
wmap = dict(zip(pool.RID[obs], w[obs])); cohi = coh.set_index('RID')
def load(run):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == 'full']; return d.groupby('rid').prob.mean()
runs = {'LR clinical+fs29': 'lr_full_fs29', 'LR clinical only': 'lr_full_clinonly', 'LR no-circular clinical+fs29': 'lr_nocirc_fs29', 'Attention clinical+fs29': 'attn_full_fs29'}
P = {k: load(v) for k, v in runs.items()}; ids = [r for r in P['LR clinical+fs29'].index if r in wmap]
y = cohi.loc[ids, 'label_primary'].astype(int).values; wt = np.array([wmap[r] for r in ids]); PP = {k: v.loc[ids].values for k, v in P.items()}
print(f'\ndev participants with weights: {len(ids)} | progression prevalence unweighted {y.mean():.3f}, weighted {np.average(y, weights=wt):.3f}')
rng = np.random.default_rng(0); B = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if 0 < y[i].sum() < len(y)]
out = []
for k, p in PP.items():
    bs = [AUC(y[i], p[i], sample_weight=wt[i]) for i in B]
    out.append(dict(model=k, auroc_unweighted=AUC(y, p), auroc_weighted=AUC(y, p, sample_weight=wt), w_ci_lo=np.percentile(bs, 2.5), w_ci_hi=np.percentile(bs, 97.5),
                    prauc_unweighted=AP(y, p), prauc_weighted=AP(y, p, sample_weight=wt), brier_weighted=np.average((p - y) ** 2, weights=wt)))
print(pd.DataFrame(out).round(3).to_string(index=False))
a, b = PP['LR clinical+fs29'], PP['LR clinical only']
d = [AUC(y[i], a[i], sample_weight=wt[i]) - AUC(y[i], b[i], sample_weight=wt[i]) for i in B]
print(f"FreeSurfer increment: weighted {AUC(y, a, sample_weight=wt) - AUC(y, b, sample_weight=wt):+.3f} {np.percentile(d, [2.5, 97.5]).round(3)} | unweighted {AUC(y, a) - AUC(y, b):+.3f}")
pd.DataFrame(out).to_csv('results/ipcw_dev_summary.csv', index=False)