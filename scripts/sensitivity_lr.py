import re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP

coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
C = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']
Xc = cl.reindex(coh.RID)[C].astype(float).values
fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
R = fsd.reindex(coh.RID); Xf = R[fcols].astype(float).copy()
if 'ST10CV' in R:
    for c in fcols:
        if c.endswith('SV'): Xf[c] = Xf[c] / R['ST10CV'] * 1000
Xf = Xf.values
mfs0 = (ms.reindex(coh.RID)['freesurfer_status'].isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(Xf).all(1)).astype(np.float32)
fsday = pd.read_csv('data/processed/features/freesurfer_raw_v1.csv').set_index('RID')['fs_days_from_anchor'].reindex(coh.RID).values
rev = coh.reverter_flag.astype(str).str.lower().eq('true').values
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']
idx = {r: i for i, r in enumerate(coh.RID)}

def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        yield np.array([idx[r] for r in g.rid.values]), np.array([idx[r] for r in dev.rid.unique() if r not in set(g.rid.values)])

def lr(X, y, Xv):
    return LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(X, y).predict_proba(Xv)[:, 1]

def run(y, keep, mfs):
    use = keep & ~np.isnan(y); P = {'clin': {}, 'full': {}}
    for rep in sorted(dev.repeat.unique()):
        for val, trn in folds(rep):
            trn, val = trn[use[trn]], val[use[val]]
            med = np.nanmedian(Xc[trn], 0); Zc = np.where(np.isnan(Xc), med, Xc); Zc = (Zc - Zc[trn].mean(0)) / (Zc[trn].std(0) + 1e-6)
            t = trn[mfs[trn] > 0]; mf = np.nanmedian(Xf[t], 0); Zf = np.where(np.isnan(Xf), mf, Xf)
            Zf = (Zf - Zf[t].mean(0)) / (Zf[t].std(0) + 1e-6); Zf[mfs == 0] = 0
            A = np.hstack([Zc, Zf, mfs[:, None]])
            for k, p in (('clin', lr(Zc[trn], y[trn], Zc[val])), ('full', lr(A[trn], y[trn], A[val]))):
                for i, v in zip(val, p): P[k].setdefault(i, []).append(v)
    ids = np.array(sorted(P['full'])); yy = y[ids].astype(int)
    return yy, np.array([np.mean(P['clin'][i]) for i in ids]), np.array([np.mean(P['full'][i]) for i in ids])

rng = np.random.default_rng(0); out = []
allt = np.ones(len(coh), bool); ph = coh.adni_phase.values
V = {'primary': (coh.label_primary.values, allt, mfs0),
     'strict_pre_anchor_FS': (coh.label_primary.values, allt, mfs0 * (fsday <= 0)),
     'excl_early_conv(<=180d)': (coh.label_excl_early.values if 'label_excl_early' in coh else None, allt, mfs0),
     'AD_to_820d': (coh.label_sens_ad820.values if 'label_sens_ad820' in coh else None, allt, mfs0),
     'neg_window_640-820': (coh.label_sens_symm90.values if 'label_sens_symm90' in coh else None, allt, mfs0),
     'neg_window_550-910': (coh.label_sens_symm180.values if 'label_sens_symm180' in coh else None, allt, mfs0),
     'neg_visit_ge730': (coh.label_sens_ge730.values if 'label_sens_ge730' in coh else None, allt, mfs0),
     'no_reverters': (coh.label_primary.values, ~rev, mfs0),
     'ADNI1_only': (coh.label_primary.values, ph == 'ADNI1', mfs0),
     'excl_ADNI1': (coh.label_primary.values, ph != 'ADNI1', mfs0)}
for name, (y, keep, mfs) in V.items():
    if y is None: print('skip (column missing):', name); continue
    yy, pc, pf = run(y.astype(float), keep, mfs.astype(np.float32))
    ba, bd = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yy), len(yy))
        if 0 < yy[i].sum() < len(i): ba.append(AUC(yy[i], pf[i])); bd.append(AUC(yy[i], pf[i]) - AUC(yy[i], pc[i]))
    out.append(dict(variant=name, n=len(yy), pos=int(yy.sum()), auc_clin=AUC(yy, pc), auc_clin_fs=AUC(yy, pf),
                    ci_lo=np.percentile(ba, 2.5), ci_hi=np.percentile(ba, 97.5), delta_fs=AUC(yy, pf) - AUC(yy, pc),
                    d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5), prauc_clin_fs=AP(yy, pf)))
    print(pd.DataFrame(out[-1:]).round(3).to_string(header=(len(out) == 1), index=False), flush=True)
pd.DataFrame(out).to_csv('results/sensitivity_lr.csv', index=False)