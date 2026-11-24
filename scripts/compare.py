import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
def load(p, c):
    d = pd.read_csv(p); d = d[d.cond == c]; return d.groupby('rid').prob.mean()
A, B = load(sys.argv[1], sys.argv[2]), load(sys.argv[3], sys.argv[4])
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID')
ids = A.index.intersection(B.index); y = coh.loc[ids, 'label_primary'].astype(int).values
a, b = A.loc[ids].values, B.loc[ids].values
rng = np.random.default_rng(0); d1, d2 = [], []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(i):
        d1.append(AUC(y[i], a[i]) - AUC(y[i], b[i])); d2.append(AP(y[i], a[i]) - AP(y[i], b[i]))
print(f'n={len(y)}  dAUROC={AUC(y,a)-AUC(y,b):+.3f} CI {np.percentile(d1,[2.5,97.5]).round(3)}  dPRAUC={AP(y,a)-AP(y,b):+.3f} CI {np.percentile(d2,[2.5,97.5]).round(3)}')