import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as AUC
os.makedirs('results/paper', exist_ok=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); Y = coh[coh.label_primary.notna()].set_index('RID').label_primary.astype(int)
def load(run, cond='full'):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == cond]; return d.groupby('rid').prob.mean()
S = {'Switch: LR full / LR clinical-only if FS missing': (load('lr_full_fs29'), load('lr_full_clinonly')),
     'LR zero-fill + indicator': (load('lr_full_fs29'), load('lr_full_fs29', 'nofs')),
     'Concat + mask': (load('concat_full_fs29'), load('concat_full_fs29', 'nofs')),
     'Concat + mask + modality dropout': (load('concatdrop_full_fs29'), load('concatdrop_full_fs29', 'nofs')),
     'Attention + mask': (load('attn_full_fs29'), load('attn_full_fs29', 'nofs')),
     'Attention + mask + modality dropout': (load('attndrop_full_fs29'), load('attndrop_full_fs29', 'nofs'))}
ids = sorted(set.intersection(*[set(a.index) & set(b.index) for a, b in S.values()]) & set(Y.index)); y = Y.loc[ids].values
PS = [0, .1, .25, .5, .75, 1.0]; N = 300; rng = np.random.default_rng(0)
masks = {p: rng.random((N, len(ids))) < p for p in PS}; rows = []
for name, (f, m) in S.items():
    f_, m_ = f.loc[ids].values, m.loc[ids].values
    for p in PS:
        a = [AUC(y, np.where(mk, m_, f_)) for mk in masks[p]]; rows.append(dict(strategy=name, p_fs_missing=p, auroc=np.mean(a), sd=np.std(a)))
T = pd.DataFrame(rows); T.to_csv('results/paper/T9_missingness_curve.csv', index=False)
print('n =', len(ids), '| positives =', int(y.sum()))
print(T.pivot(index='strategy', columns='p_fs_missing', values='auroc').round(3).to_string())
ref = T[T.strategy.str.startswith('Switch')].set_index('p_fs_missing').auroc
print('\nDifference vs the Switch fallback (negative = worse than the simple fallback):')
print(T.assign(diff=lambda d: d.auroc - d.p_fs_missing.map(ref)).pivot(index='strategy', columns='p_fs_missing', values='diff').round(3).to_string())
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name in S: t = T[T.strategy == name]; ax.plot(t.p_fs_missing, t.auroc, 'o-', label=name)
ax.set_xlabel('fraction of participants with FreeSurfer masked at test time'); ax.set_ylabel('AUROC (dev out-of-fold)'); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig('results/paper/F_missingness_curve.png', dpi=130)