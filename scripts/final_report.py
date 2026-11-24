import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR, roc_curve
M = {'final_lr_full_fs29': ('lr_full_fs29', 'LR clinical+fs29 (PRIMARY)'),
     'final_lr_full_clinonly': ('lr_full_clinonly', 'LR clinical only'),
     'final_lr_nocirc_fs29': ('lr_nocirc_fs29', 'LR no-circular clinical+fs29'),
     'final_attn_full_fs29': ('attn_full_fs29', 'Attention clinical+fs29'),
     'final_concat_full_fs29': ('concat_full_fs29', 'Concat clinical+fs29'),
     'final_mri_fs29_drop': ('mri_fs29_drop', 'Attention + MRI (dropout 0.3)')}
PRIM = 'final_lr_full_fs29'; THR = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]; ALPHA = 0.10
os.makedirs('results/final', exist_ok=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID')
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
T, D, S = {}, {}, {}
for k, (cv, _) in M.items():
    t = pd.read_csv(f'results/deep/{k}/oof.csv'); t = t[t.cond == 'full']; T[k] = t.groupby('rid').prob.mean()
    S[k] = [AUC(coh.loc[g.rid, 'label_primary'].astype(int), g.prob) for _, g in t.groupby('repeat')]
    d = pd.read_csv(f'results/deep/{cv}/oof.csv'); d = d[d.cond == 'full']; D[k] = d.groupby('rid').prob.mean()
ids = sorted(set.intersection(*[set(v.index) for v in T.values()]))
y = coh.loc[ids, 'label_primary'].astype(int).values; n = len(y); print('locked test n =', n, 'positives =', y.sum())
rng = np.random.default_rng(0); B = [i for i in (rng.integers(0, n, n) for _ in range(2000)) if 0 < y[i].sum() < n]
def ece(p, yy):
    g = pd.DataFrame({'p': p, 'y': yy, 'q': pd.qcut(p, 10, duplicates='drop')}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())
P, R, cal = {}, {}, {}
for k in M:
    p = T[k].loc[ids].values; yd = coh.loc[D[k].index, 'label_primary'].astype(int).values
    cal[k] = LogisticRegression(C=1e6).fit(lg(D[k].values)[:, None], yd); P[k] = p; R[k] = cal[k].predict_proba(lg(p)[:, None])[:, 1]
rows = []
for k, (_, lab) in M.items():
    p, r = P[k], R[k]; ci = lambda f, x: np.percentile([f(y[i], x[i]) for i in B], [2.5, 97.5]).round(3)
    sl = LogisticRegression(C=1e6).fit(lg(p)[:, None], y).coef_[0, 0]
    rows.append(dict(model=lab, auroc=AUC(y, p), auroc_ci=str(ci(AUC, p)), prauc=AP(y, p), prauc_ci=str(ci(AP, p)),
                     brier_raw=BR(y, p), brier_recal=BR(y, r), ece_recal=ece(r, y), test_cal_slope_raw=sl,
                     seed_auroc_mean=np.mean(S[k]), seed_auroc_sd=np.std(S[k]), n_seeds=len(S[k])))
pd.DataFrame(rows).round(3).to_csv('results/final/final_metrics.csv', index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
dl = []
for k, (_, lab) in M.items():
    if k == PRIM: continue
    da = [AUC(y[i], P[PRIM][i]) - AUC(y[i], P[k][i]) for i in B]; dp = [AP(y[i], P[PRIM][i]) - AP(y[i], P[k][i]) for i in B]
    dl.append(dict(primary_minus=lab, dAUROC=AUC(y, P[PRIM]) - AUC(y, P[k]), lo=np.percentile(da, 2.5), hi=np.percentile(da, 97.5),
                   dPRAUC=AP(y, P[PRIM]) - AP(y, P[k]), plo=np.percentile(dp, 2.5), phi=np.percentile(dp, 97.5)))
pd.DataFrame(dl).round(3).to_csv('results/final/final_deltas.csv', index=False); print('\nPaired differences (primary minus other):'); print(pd.DataFrame(dl).round(3).to_string(index=False))
r = R[PRIM]; th = []
for t in THR:
    pr = r >= t; tp, fp, fn, tn = (pr & (y == 1)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum(), (~pr & (y == 0)).sum()
    nb = lambda yy, rr: ((rr >= t) & (yy == 1)).sum() / len(yy) - ((rr >= t) & (yy == 0)).sum() / len(yy) * t / (1 - t)
    nbc = np.percentile([nb(y[i], r[i]) for i in B], [2.5, 97.5])
    th.append(dict(thr=t, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), flagged=pr.mean(),
                   net_benefit=nb(y, r), nb_lo=nbc[0], nb_hi=nbc[1], treat_all=y.mean() - (1 - y.mean()) * t / (1 - t)))
pd.DataFrame(th).round(3).to_csv('results/final/final_thresholds_primary.csv', index=False); print('\nPrimary model (dev-recalibrated):'); print(pd.DataFrame(th).round(3).to_string(index=False))
ph = coh.loc[ids, 'adni_phase'].values; pg = np.where(ph == 'ADNI1', 'ADNI1', np.where(np.isin(ph, ['ADNIGO', 'ADNI2']), 'ADNIGO+2', 'ADNI3'))
print('\nPrimary AUROC by phase group (descriptive):')
for g in ['ADNI1', 'ADNIGO+2', 'ADNI3']:
    m = pg == g; print(g, 'n=', m.sum(), 'pos=', y[m].sum(), 'AUROC=', round(AUC(y[m], P[PRIM][m]), 3) if 0 < y[m].sum() < m.sum() else 'NA')
from sklearn.model_selection import train_test_split

dv = D[PRIM]
yd = coh.loc[dv.index, 'label_primary'].astype(int).values
fit_idx, conf_idx = train_test_split(
    np.arange(len(dv)), test_size=0.5, random_state=42, stratify=yd
)

# Fit calibration on one development subset only.
conf_cal = LogisticRegression(C=1e6).fit(
    lg(dv.values[fit_idx])[:, None], yd[fit_idx]
)

# Estimate class-conditional conformal quantiles on the disjoint subset.
p_conf = conf_cal.predict_proba(lg(dv.values[conf_idx])[:, None])[:, 1]
y_conf = yd[conf_idx]
q = {}
for c in (0, 1):
    scores = (1 - p_conf if c == 1 else p_conf)[y_conf == c]
    k = int(np.ceil((len(scores) + 1) * (1 - ALPHA)))
    q[c] = np.sort(scores)[min(k, len(scores)) - 1]

# Apply the independently calibrated predictor to locked-test probabilities.
r_conf = conf_cal.predict_proba(lg(P[PRIM])[:, None])[:, 1]
in1, in0 = (1 - r_conf) <= q[1], r_conf <= q[0]
sing1, sing0 = in1 & ~in0, in0 & ~in1
both, none = in1 & in0, ~in1 & ~in0
print(f'\nClass-conditional split conformal, alpha={ALPHA}: coverage class1={in1[y==1].mean():.3f} class0={in0[y==0].mean():.3f}')
print(f'singleton={(sing1|sing0).mean():.3f} both-classes(abstain)={both.mean():.3f} empty={none.mean():.3f}')
print(f'on confident predictions: PPV(pred 1)={y[sing1].mean() if sing1.any() else float("nan"):.3f} (n={sing1.sum()})  NPV(pred 0)={1-y[sing0].mean() if sing0.any() else float("nan"):.3f} (n={sing0.sum()})')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, (_, lab) in M.items(): f, t_, _ = roc_curve(y, P[k]); ax[0].plot(f, t_, label=f'{lab} {AUC(y, P[k]):.2f}')
ax[0].plot([0, 1], [0, 1], 'k--'); ax[0].legend(fontsize=6); ax[0].set_title('ROC (locked test)')
g = pd.DataFrame({'p': r, 'y': y, 'q': pd.qcut(r, 5, duplicates='drop')}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'))
ax[1].plot(g.p, g.y, 'o-'); ax[1].plot([0, 1], [0, 1], 'k--'); ax[1].set_title('Calibration, primary (5 bins)')
ts = np.linspace(0.05, 0.6, 56); nbf = lambda rr, t: ((rr >= t) & (y == 1)).sum() / n - ((rr >= t) & (y == 0)).sum() / n * t / (1 - t)
ax[2].plot(ts, [nbf(R[PRIM], t) for t in ts], label='primary'); ax[2].plot(ts, [nbf(R['final_lr_full_clinonly'], t) for t in ts], label='clinical only')
ax[2].plot(ts, [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ts], label='treat all'); ax[2].axhline(0, color='k'); ax[2].set_ylim(-0.05, 0.25); ax[2].legend(); ax[2].set_title('Decision curve')
plt.tight_layout(); plt.savefig('results/final/final_figures.png', dpi=110)