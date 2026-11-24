import re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True); coh['y'] = coh.label_primary.astype(int)
sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid'); coh['dev'] = (coh.RID.map(sp.split) != 'test').values
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(coh.RID)
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
coh['fs'] = ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values; coh['mri'] = coh.RID.isin(ok).values
coh['grp'] = np.where(coh.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3')
coh['pattern'] = np.select([coh.fs & coh.mri, coh.fs, coh.mri], ['clin+FS+MRI', 'clin+FS', 'clin+MRI'], 'clin only')
print('=== 1. Real availability patterns ===')
for g, c in [('all 778', coh), ('development 622', coh[coh.dev])]:
    print(f'\n[{g}]'); print(c.groupby(['pattern', 'grp']).y.agg(n='size', pos='sum').assign(rate=lambda x: (x.pos / x.n).round(3)).to_string())
def load(run, cond='full'):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == cond]; return d.groupby('rid').prob.mean()
SW = 'Switch'
S = {SW: (load('lr_full_fs29'), load('lr_full_clinonly')), 'LR zero-fill': (load('lr_full_fs29'), load('lr_full_fs29', 'nofs')),
     'Concat+mask': (load('concat_full_fs29'), load('concat_full_fs29', 'nofs')), 'Attention+mask+dropout': (load('attndrop_full_fs29'), load('attndrop_full_fs29', 'nofs'))}
ids = sorted(set.intersection(*[set(a.index) & set(b.index) for a, b in S.values()]) & set(coh.RID)); c2 = coh.set_index('RID').loc[ids]; y = c2.y.values; grp = c2.grp.values
fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
hv = ((fsd.ST29SV + fsd.ST88SV) / fsd.ST10CV).reindex(ids); hv = hv.fillna(hv.median()).values
F = {k: (f.loc[ids].values, m.loc[ids].values) for k, (f, m) in S.items()}
scen = {'all FS missing': np.ones(len(ids), bool), 'lowest-25% hippocampal volume': hv <= np.percentile(hv, 25)}
print('\n=== 2. Phase-stratified FreeSurfer-loss scenarios (development, post hoc; paired participant bootstrap) ===')
for g in ['ADNI1', 'ADNIGO/2/3']:
    sel = grp == g; yy = y[sel]; B = []
    while len(B) < 1000:
        i = rng.integers(0, sel.sum(), sel.sum())
        if 0 < yy[i].sum() < len(i): B.append(i)
    for sc, mk in scen.items():
        P = {k: np.where(mk, F[k][1], F[k][0])[sel] for k in S}
        for k in S:
            dd = [AUC(yy[i], P[k][i]) - AUC(yy[i], P[SW][i]) for i in B]
            print(f'{g:11s} n={sel.sum()} pos={yy.sum()} | {sc:30s} | {k:24s} AUROC {AUC(yy, P[k]):.3f}  diff vs switch {AUC(yy, P[k]) - AUC(yy, P[SW]):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]')
print('\n=== 3. Do the availability indicators carry phase or outcome information? (development, 3x5 CV, post hoc) ===')
dv = coh[coh.dev].reset_index(drop=True); yv = dv.y.values; a1 = (dv.adni_phase == 'ADNI1').astype(int).values
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(dv.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; R = fsd.reindex(dv.RID); Xf = R[fcols].astype(float).copy()
for c in fcols:
    if c.endswith('SV'): Xf[c] = Xf[c] / R['ST10CV'] * 1000
Xf = Xf.values; Xf[~dv.fs.values] = np.nan
fsi, mri = dv.fs.values.astype(float)[:, None], dv.mri.values.astype(float)[:, None]
def cvpred(X, t):
    pr = np.zeros((3, len(t))); sk = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for i, (tr, va) in enumerate(sk.split(X, t)):
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000))
        pr[i // 5, va] = m.fit(X[tr], t[tr]).predict_proba(X[va])[:, 1]
    return pr.mean(0)
mk2 = np.hstack([fsi, mri])
print('availability bits (FS usable, MRI usable) -> ADNI1: AUROC', round(AUC(a1, cvpred(mk2, a1)), 3))
print('availability bits -> outcome: AUROC', round(AUC(yv, cvpred(mk2, yv)), 3), '(phase-only reference about 0.70)')
base = np.hstack([Xc, Xf, fsi]); plus = np.hstack([base, mri]); pb, pp = cvpred(base, yv), cvpred(plus, yv); B = []
while len(B) < 2000:
    i = rng.integers(0, len(yv), len(yv))
    if 0 < yv[i].sum() < len(i): B.append(i)
dd = [AUC(yv[i], pp[i]) - AUC(yv[i], pb[i]) for i in B]
print(f'clinical+fs29+FS bit AUROC {AUC(yv, pb):.3f} | adding MRI-usable bit {AUC(yv, pp):.3f} | difference {AUC(yv, pp) - AUC(yv, pb):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]')