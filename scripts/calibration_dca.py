import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC, brier_score_loss
run = sys.argv[1] if len(sys.argv) > 1 else 'lr_full_fs29'
d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == 'full']
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID')
p = d.groupby('rid').prob.mean(); y = coh.loc[p.index, 'label_primary'].astype(int).values; p = p.values; n = len(y)
pc = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc))
cal = LogisticRegression(C=1e6).fit(lg[:, None], y)
print(f'n={n} prev={y.mean():.3f} mean_pred={p.mean():.3f} AUROC={AUC(y,p):.3f} Brier={brier_score_loss(y,p):.3f} cal_slope={cal.coef_[0,0]:.2f} cal_intercept={cal.intercept_[0]:.2f}')
g = pd.DataFrame({'p': p, 'y': y, 'q': pd.qcut(p, 10, duplicates='drop')}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
print('ECE (10 equal-frequency bins):', round(float((g.n * (g.p - g.y).abs()).sum() / g.n.sum()), 3))
rng = np.random.default_rng(0)
def bs(f):
    v = []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: v.append(f(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5]).round(3)
print('AUROC 95% CI', bs(AUC), '| Brier 95% CI', bs(brier_score_loss))
rows = []
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    pr = p >= t; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum(); tn = (~pr & (y == 0)).sum()
    rows.append(dict(thr=t, sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), flagged=pr.mean(),
                     net_benefit=tp / n - fp / n * t / (1 - t), treat_all=y.mean() - (1 - y.mean()) * t / (1 - t)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(g.p, g.y, 'o-'); ax[0].plot([0, 1], [0, 1], 'k--'); ax[0].set_xlabel('predicted'); ax[0].set_ylabel('observed')
ts = np.linspace(0.05, 0.6, 56)
ax[1].plot(ts, [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in ts], label=run)
ax[1].plot(ts, [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ts], label='treat all'); ax[1].axhline(0, color='k', label='treat none')
ax[1].set_ylim(-0.05, 0.25); ax[1].legend(); ax[1].set_xlabel('threshold')
plt.tight_layout(); plt.savefig(f'results/calib_dca_{run}.png', dpi=100)