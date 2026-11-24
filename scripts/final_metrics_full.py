import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR, log_loss as LL
M = {'final_lr_full_fs29': ('lr_full_fs29', 'LR clinical+fs29 (primary)'),
     'final_lr_full_clinonly': ('lr_full_clinonly', 'LR clinical only'),
     'final_lr_nocirc_fs29': ('lr_nocirc_fs29', 'LR no-circular clinical+fs29'),
     'final_attn_full_fs29': ('attn_full_fs29', 'Attention clinical+fs29'),
     'final_concat_full_fs29': ('concat_full_fs29', 'Concat clinical+fs29'),
     'final_mri_fs29_drop': ('mri_fs29_drop', 'Attention + MRI (dropout 0.3)')}
os.makedirs('results/final', exist_ok=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID')
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def load(path):
    d = pd.read_csv(path); d = d[d.cond == 'full']; return d.groupby('rid').prob.mean()
def ece(p, yy):
    g = pd.DataFrame({'p': p, 'y': yy, 'q': pd.qcut(p, 10, duplicates='drop')}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())
def at(y, r, t):
    pr = r >= t; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum(); tn = (~pr & (y == 0)).sum()
    se, sp, pv, nv = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    return dict(accuracy=(tp + tn) / len(y), balanced_acc=(se + sp) / 2, precision=pv, recall=se, specificity=sp, f1=2 * pv * se / max(pv + se, 1e-12), npv=nv), (tp, fp, fn, tn)
rng = np.random.default_rng(0)
tests = {k: load(f'results/deep/{k}/oof.csv') for k in M}; devs = {k: load(f'results/deep/{v[0]}/oof.csv') for k, v in M.items()}
ids = sorted(set.intersection(*[set(t.index) for t in tests.values()]))
yt = coh.loc[ids, 'label_primary'].astype(int).values; n = len(yt); print('locked test n =', n, 'positives =', yt.sum())
B = [i for i in (rng.integers(0, n, n) for _ in range(2000)) if 0 < yt[i].sum() < n]
ci = lambda a: np.percentile(a, [2.5, 97.5]); fmt = lambda v, lo, hi: f'{v:.3f} [{lo:.3f}, {hi:.3f}]'
grid = np.round(np.arange(0.05, 0.71, 0.01), 2); ops, gl = [], []
for k, (cv, lab) in M.items():
    dv = devs[k]; yd = coh.loc[dv.index, 'label_primary'].astype(int).values
    cal = LogisticRegression(C=1e6).fit(lg(dv.values)[:, None], yd); rd = cal.predict_proba(lg(dv.values)[:, None])[:, 1]
    p = tests[k].loc[ids].values; r = cal.predict_proba(lg(p)[:, None])[:, 1]
    t_dev = float(grid[int(np.argmax([at(yd, rd, t)[0]['balanced_acc'] for t in grid]))])
    for nm, t in [('dev_balacc_optimal', t_dev), ('fixed_0.20', 0.20), ('fixed_0.30', 0.30)]:
        m, c = at(yt, r, t); bs = pd.DataFrame([at(yt[i], r[i], t)[0] for i in B])
        row = dict(model=lab, operating_point=nm, thr=t, TP=c[0], FP=c[1], FN=c[2], TN=c[3])
        row.update({mn: fmt(m[mn], *ci(bs[mn])) for mn in m}); ops.append(row)
    fit = lambda yy, pp: LogisticRegression(C=1e6).fit(lg(pp)[:, None], yy)
    f0 = fit(yt, r); bsl = np.array([(lambda f: (f.coef_[0, 0], f.intercept_[0]))(fit(yt[i], r[i])) for i in B[:500]])
    gl.append(dict(model=lab, auroc=fmt(AUC(yt, p), *ci([AUC(yt[i], p[i]) for i in B])), prauc=fmt(AP(yt, p), *ci([AP(yt[i], p[i]) for i in B])),
                   brier=fmt(BR(yt, r), *ci([BR(yt[i], r[i]) for i in B])), logloss=fmt(LL(yt, np.clip(r, 1e-6, 1 - 1e-6)), *ci([LL(yt[i], np.clip(r[i], 1e-6, 1 - 1e-6), labels=[0, 1]) for i in B])),
                   ece=round(ece(r, yt), 3), cal_slope=fmt(f0.coef_[0, 0], *ci(bsl[:, 0])), cal_intercept=fmt(f0.intercept_[0], *ci(bsl[:, 1]))))
pd.DataFrame(ops).to_csv('results/final/final_full_metrics_operating.csv', index=False); pd.DataFrame(gl).to_csv('results/final/final_full_metrics_global.csv', index=False)
print('\nGLOBAL (Brier, log loss, ECE, slope use dev-recalibrated probabilities):'); print(pd.DataFrame(gl).to_string(index=False))
print('\nOPERATING POINTS (threshold from dev only, applied to locked test):'); print(pd.DataFrame(ops).to_string(index=False))
P0, C0 = 'final_lr_full_fs29', 'final_lr_full_clinonly'
def pooled(k): yd = coh.loc[devs[k].index, 'label_primary'].astype(int).values; return np.r_[yd, yt], np.r_[devs[k].values, tests[k].loc[ids].values]
yP, pP = pooled(P0); _, pC = pooled(C0); N = len(yP)
BP = [i for i in (rng.integers(0, N, N) for _ in range(2000)) if 0 < yP[i].sum() < N]
d = [AUC(yP[i], pP[i]) - AUC(yP[i], pC[i]) for i in BP]
print(f'\nPOOLED dev-OOF + locked test (n={N}, pos={yP.sum()}), DESCRIPTIVE: primary AUROC {AUC(yP, pP):.3f} {ci([AUC(yP[i], pP[i]) for i in BP]).round(3)}; clinical-only {AUC(yP, pC):.3f}; FS increment {AUC(yP, pP) - AUC(yP, pC):+.3f} {ci(d).round(3)}')