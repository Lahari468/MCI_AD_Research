import glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
R = 'results/experiments/'
oof = pd.read_csv(R + 'e1_oof.csv'); print('e1_oof columns:', list(oof.columns))
oof = oof.rename(columns={next(c for c in oof.columns if k in c.lower()): k for k in ['model', 'repeat', 'rid', 'prob']})
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID'); sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid'); test = set(sp[sp.split == 'test'].index)
print('\n=== 1. structure ===')
print('models', oof.model.nunique(), '| repeats', oof['repeat'].nunique(), '| participants', oof.rid.nunique(), '| rows', len(oof), '| expected', oof.model.nunique() * oof['repeat'].nunique() * 564)
print('NaN probs', int(oof.prob.isna().sum()), '| test RIDs present', len(set(oof.rid) & test), '| duplicate (model, repeat, rid)', int(oof.duplicated(['model', 'repeat', 'rid']).sum()))
rid = np.array(sorted(oof.rid.unique())); y = coh.loc[rid, 'label_primary'].astype(int).values; g1 = (coh.loc[rid, 'adni_phase'] == 'ADNI1').values
print('participants', len(rid), '| events', int(y.sum()), '| ADNI1', int(g1.sum()), '| later', int((~g1).sum()))
fo = pd.read_csv(R + 'e1_folds.csv'); print('folds file columns', list(fo.columns), '| rows', len(fo), '| test RIDs in folds', len(set(fo[[c for c in fo.columns if 'rid' in c.lower()][0]]) & test))
M = {m: d.pivot(index='rid', columns='repeat', values='prob').loc[rid] for m, d in oof.groupby('model')}; pool = {m: v.mean(1).values for m, v in M.items()}
rep = pd.read_csv(R + 'e1_summary.csv'); mp = rep[rep.section == 'model_performance'].set_index('item')
print('\n=== 2. model AUROCs: recomputed | reported ===')
for m in M:
    pr = [AUC(y, M[m][c].values) for c in M[m].columns]
    print(f'{m:15s} pooled {AUC(y, pool[m]):.4f} | {mp.loc[m, "pooled_estimate"]:.4f}   repeat mean {np.mean(pr):.4f} (sd0 {np.std(pr):.4f}, sd1 {np.std(pr, ddof=1):.4f}) | {mp.loc[m, "repeat_mean"]:.4f} +- {mp.loc[m, "repeat_sd"]:.4f}   ADNI1 {AUC(y[g1], pool[m][g1]):.4f} | {mp.loc[m, "adni1_estimate"]:.4f}   later {AUC(y[~g1], pool[m][~g1]):.4f} | {mp.loc[m, "later_estimate"]:.4f}')
rng = np.random.default_rng(12345); B = []
while len(B) < 2000:
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(y): B.append(i)
est = rep[rep.section == 'estimand_contrast'].set_index('item'); print('\n=== 3. contrasts: recomputed (fresh bootstrap) | reported ===')
for k, r in est.iterrows():
    a, b = [s.strip() for s in k.split(' - ')]; d = [AUC(y[i], pool[a][i]) - AUC(y[i], pool[b][i]) for i in B]
    pe = AUC(y, pool[a]) - AUC(y, pool[b]); per = [AUC(y, M[a][c].values) - AUC(y, M[b][c].values) for c in M[a].columns]
    print(f'{k:32s} {pe:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] | {r.pooled_estimate:+.4f} [{r.ci95_lower:+.4f}, {r.ci95_upper:+.4f}]  per-repeat {np.mean(per):+.4f} | {r.repeat_mean:+.4f}')
print('\n=== 4. permutation nulls ===')
obs = {n: est.loc[n, 'pooled_estimate'] for n in ['C+ROI18 - C', 'C+prot+ROI18 - C+prot']}
for f in sorted(glob.glob(R + 'e1_permutation_null_*.npy')):
    x = np.load(f).ravel(); print(f.split('/')[-1], 'n', len(x), 'mean', round(x.mean(), 4), 'sd', round(x.std(), 4), 'max', round(x.max(), 4), {n: round((1 + (x >= o).sum()) / (len(x) + 1), 4) for n, o in obs.items()})
print(rep[rep.section == 'permutation_test'][['item', 'pooled_estimate', 'perm_B', 'perm_p_value']].to_string(index=False))
print('\n=== 5. E2 diagnostic: mixture gain vs within-group gain (saved fallback predictions only) ===')
fb = pd.read_csv('results/mri_fallback/oof.csv'); print('fallback columns', list(fb.columns), '| models', sorted(fb.model.unique()))
pf = fb.groupby(['model', 'rid']).prob.mean().unstack(0).reindex(rid); F, C, CR = pf['F'].values, pf['C'].values, pf['C+ROI'].values
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID'); real = ms.reindex(rid).freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values
print('participants', len(rid), '| real FreeSurfer-unavailable', int((~real).sum()), '| events among them', int(y[~real].sum()), '| mean predicted prob F/C/C+ROI', round(F.mean(), 4), round(C.mean(), 4), round(CR.mean(), 4), '| prevalence', round(y.mean(), 4))
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rg = np.random.default_rng(7); out = []
    for _ in range(200):
        un = (~real) | (rg.random(len(rid)) < p)
        if un.sum() < 20 or y[un].sum() < 5 or (y[un] == 0).sum() < 5: continue
        out.append((AUC(y, np.where(un, CR, F)) - AUC(y, np.where(un, C, F)), AUC(y[un], CR[un]) - AUC(y[un], C[un]), un.mean()))
    o = np.array(out); print(f'p={p:4.2f}  masks {len(o):3d}  unavailable share {o[:, 2].mean():.2f}  mixture gain {o[:, 0].mean():+.4f}  within-unavailable gain {o[:, 1].mean():+.4f}  excess {o[:, 0].mean() - o[:, 1].mean():+.4f}')