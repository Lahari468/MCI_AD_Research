import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
pa, ca, pb, cb = sys.argv[1:5]
subset = sys.argv[5] if len(sys.argv) > 5 else 'all'      # all | mri | nomri
rep = int(sys.argv[6]) if len(sys.argv) > 6 else -1
def load(p, c):
    d = pd.read_csv(p); d = d[d.cond == c]
    if rep >= 0: d = d[d.repeat == rep]
    return d.groupby('rid').prob.mean()
A, B = load(pa, ca), load(pb, cb)
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
ok = set(log[log.status.str.startswith(('OK', 'REVIEW'))].rid)
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID')
ids = A.index.intersection(B.index)
if subset == 'mri': ids = [i for i in ids if i in ok]
if subset == 'nomri': ids = [i for i in ids if i not in ok]
y = coh.loc[ids, 'label_primary'].astype(int).values; a, b = A.loc[ids].values, B.loc[ids].values
rng = np.random.default_rng(0); d1, d2 = [], []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(i): d1.append(AUC(y[i], a[i]) - AUC(y[i], b[i])); d2.append(AP(y[i], a[i]) - AP(y[i], b[i]))
print(f'subset={subset} n={len(y)} pos={y.sum()} AUROC A={AUC(y,a):.3f} B={AUC(y,b):.3f}  dAUROC={AUC(y,a)-AUC(y,b):+.3f} CI {np.percentile(d1,[2.5,97.5]).round(3)}  dPRAUC={AP(y,a)-AP(y,b):+.3f} CI {np.percentile(d2,[2.5,97.5]).round(3)}')