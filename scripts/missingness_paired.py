import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
coh = pd.read_csv('data/processed/cohort_v2b.csv'); Y = coh[coh.label_primary.notna()].set_index('RID').label_primary.astype(int)
def load(run, cond='full', rep=None):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == cond]
    if rep is not None: d = d[d.repeat == rep]
    return d.groupby('rid').prob.mean()
SW = 'Switch (LR full / LR clinical-only)'
S = {SW: (load('lr_full_fs29'), load('lr_full_clinonly')),
     'LR zero-fill + indicator': (load('lr_full_fs29'), load('lr_full_fs29', 'nofs')),
     'Concat + mask': (load('concat_full_fs29'), load('concat_full_fs29', 'nofs')),
     'Concat + mask + dropout': (load('concatdrop_full_fs29'), load('concatdrop_full_fs29', 'nofs')),
     'Attention + mask': (load('attn_full_fs29'), load('attn_full_fs29', 'nofs')),
     'Attention + mask + dropout': (load('attndrop_full_fs29'), load('attndrop_full_fs29', 'nofs'))}
ids = sorted(set.intersection(*[set(a.index) & set(b.index) for a, b in S.values()]) & set(Y.index)); y = Y.loc[ids].values; n = len(ids)
fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
hv = ((fsd.ST29SV + fsd.ST88SV) / fsd.ST10CV).reindex(ids); hv = hv.fillna(hv.median()).values
rng = np.random.default_rng(0)
scen = {'MCAR 50% FS missing': rng.random(n) < 0.5, 'MNAR lowest-25% hippocampal volume': hv <= np.percentile(hv, 25),
        'MNAR lowest-50% hippocampal volume': hv <= np.percentile(hv, 50), 'All FS missing': np.ones(n, bool)}
B = []
while len(B) < 2000:
    i = rng.integers(0, n, n)
    if 0 < y[i].sum() < n: B.append(i)
aucs = lambda p: np.array([AUC(y[i], p[i]) for i in B])
F = {k: (f.loc[ids].values, m.loc[ids].values) for k, (f, m) in S.items()}; fb = {k: aucs(F[k][0]) for k in S}; rows = []
for sc, mk in scen.items():
    mp = {k: np.where(mk, F[k][1], F[k][0]) for k in S}; mb = {k: aucs(mp[k]) for k in S}
    for k in S:
        pt, ps = AUC(y, mp[k]), AUC(y, mp[SW]); deg, dsw = AUC(y, F[k][0]) - pt, AUC(y, F[SW][0]) - ps
        d1 = mb[k] - mb[SW]; dg = (fb[k] - mb[k]) - (fb[SW] - mb[SW])
        rows.append(dict(scenario=sc, strategy=k, auroc=pt, d_vs_switch=pt - ps, lo=np.percentile(d1, 2.5), hi=np.percentile(d1, 97.5),
                         degradation=deg, degr_minus_switch=deg - dsw, dlo=np.percentile(dg, 2.5), dhi=np.percentile(dg, 97.5)))
T = pd.DataFrame(rows).round(3); T.to_csv('results/paper/T9b_missingness_paired.csv', index=False)
print(f'n={n} pos={y.sum()} | masked fractions: ' + ', '.join(f'{k}: {m.mean():.2f}' for k, m in scen.items())); print(T.to_string(index=False))
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid); ph = coh.set_index('RID').adni_phase
A, N0, L0 = load('mri_fs29_drop', 'full', 0), load('mri_fs29_drop', 'nomri', 0), load('lr_full_fs29', 'full', 0)
print('\nWithin ADNI1 (dev, repeat 0): MRI usable vs unusable')
for lab, flag in [('MRI usable', True), ('MRI unusable', False)]:
    m = [r for r in A.index if ph.get(r) == 'ADNI1' and ((r in ok) == flag)]; yy = Y.loc[m].values
    print(lab, 'n=', len(m), 'pos=', int(yy.sum()), '| MRI model', round(AUC(yy, A.loc[m]), 3), '| MRI masked', round(AUC(yy, N0.loc[m]), 3), '| LR clin+fs29', round(AUC(yy, L0.loc[m]), 3))