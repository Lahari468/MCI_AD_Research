import os, re, glob, numpy as np, pandas as pd, nibabel as nib
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0); FSL = os.environ['FSLDIR']; os.makedirs('results/mri_only', exist_ok=True)
img = glob.glob(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'); xml = glob.glob(f'{FSL}/data/atlases/HarvardOxford-Subcortical.xml'); assert img and xml
At = nib.load(img[0]).get_fdata().astype(int); labs = {int(i): n.strip() for i, n in re.findall(r'<label index="(\d+)"[^>]*>([^<]+)</label>', open(xml[0]).read())}
def vox(name):
    k = [i for i, n in labs.items() if n.lower() == name.lower()]; assert k, name; return At == (k[0] + 1)
ROI = {n: vox(n) for n in ['Left Hippocampus', 'Right Hippocampus', 'Left Amygdala', 'Right Amygdala', 'Left Lateral Ventricle', 'Right Lateral Ventricle']}
WM = vox('Left Cerebral White Matter') | vox('Right Cerebral White Matter')
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']; test_ids = set(sp[sp.split == 'test'].rid)
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
S = coh[coh.RID.isin(dev.rid.unique()) & coh.RID.isin(ok)]; S = S[S.RID.map(lambda r: os.path.exists(f'data/interim/mri_mni2mm_v2/{r}.npy'))].reset_index(drop=True)
assert not (set(S.RID) & test_ids), 'locked-test participant present'
y = S.label_primary.astype(int).values; n = len(S); idx = {r: i for i, r in enumerate(S.RID)}; grp = np.where(S.adni_phase == 'ADNI1', 'ADNI1', 'later'); print(f'MRI-usable development participants: {n}, events {y.sum()}')
tens = [np.load(f'data/interim/mri_mni2mm_v2/{r}.npy')[0].astype(np.float32) for r in S.RID]
def roi_feats(x):
    wm = np.median(x[WM]); out = []
    for m in ROI.values(): out += list(np.percentile(x[m], [10, 50, 90]) - wm)
    return out
RF = np.array([roi_feats(x) for x in tens]); T = np.stack([x[::4, ::4, ::4].ravel() for x in tens]); del tens
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); R = f29.reindex(S.RID)
c29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; X29 = R[c29].astype(float).copy()
for c in c29:
    if c.endswith('SV'): X29[c] = X29[c] / R['ST10CV'] * 1000
X29 = X29.values; ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(S.RID)
mfs = (ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1)).astype(float); X29[mfs == 0] = np.nan
PH = pd.get_dummies(S.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values
B = {'ph': PH, 'fs': np.hstack([X29, mfs[:, None]]), 'roi': RF, 'pc': T}
M = {'phase': ['ph'], 'fs29': ['fs'], 'PC10': ['pc'], 'ROI18': ['roi'], 'PC10+ROI18': ['pc', 'roi'], 'ph+PC10': ['ph', 'pc'], 'ph+ROI18': ['ph', 'roi'], 'ph+PC10+ROI18': ['ph', 'pc', 'roi'], 'ph+fs29': ['ph', 'fs']}
CON = [('PC10', 'phase'), ('ROI18', 'phase'), ('PC10+ROI18', 'phase'), ('fs29', 'phase'), ('PC10', 'fs29'), ('ROI18', 'fs29'), ('PC10+ROI18', 'fs29'),
       ('ph+PC10', 'phase'), ('ph+ROI18', 'phase'), ('ph+PC10+ROI18', 'phase'), ('ph+fs29', 'phase')]
def make(parts):
    arrs = [B[p] for p in parts]; X = np.hstack(arrs); steps = []; s = 0
    for p, a in zip(parts, arrs):
        cols = list(range(s, s + a.shape[1])); s += a.shape[1]
        t = Pipeline([('s', StandardScaler())]) if p == 'ph' else Pipeline([('p', PCA(10, svd_solver='randomized', random_state=0)), ('s', StandardScaler())]) if p == 'pc' else Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())])
        steps.append((p, t, cols))
    pipe = Pipeline([('pre', ColumnTransformer(steps)), ('lr', LogisticRegression(max_iter=3000))])
    return X, GridSearchCV(pipe, {'lr__C': np.logspace(-3, 1, 7)}, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = np.array(sorted(set(g.rid.values) & set(S.RID))); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(S.RID.values, v)])
reps = sorted(dev.repeat.unique()); res = {m: {r: np.full(n, np.nan) for r in reps} for m in M}
for m, parts in M.items():
    X, gs = make(parts); print('running', m, flush=True)
    for rep in reps:
        for val, trn in folds(rep): res[m][rep][val] = gs.fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
pooled = {m: np.mean([res[m][r] for r in reps], 0) for m in M}
pd.DataFrame([dict(rid=S.RID[i], repeat=r, model=m, prob=res[m][r][i]) for m in M for r in reps for i in range(n)]).to_csv('results/mri_only/oof.csv', index=False)
print('\nAUROC: pooled over repeats | per-repeat mean +- SD | ADNI1 | later')
for m in M:
    pr = [AUC(y, res[m][r]) for r in reps]; a1, a2 = (AUC(y[grp == g], pooled[m][grp == g]) for g in ('ADNI1', 'later'))
    print(f'{m:16s} {AUC(y, pooled[m]):.4f} | {np.mean(pr):.4f} +- {np.std(pr):.4f} | {a1:.4f} | {a2:.4f}')
print('\nPaired contrasts (first minus second), 95% participant bootstrap:')
for a, b in CON:
    d = []
    for _ in range(2000):
        j = rng.integers(0, n, n)
        if 0 < y[j].sum() < n: d.append(AUC(y[j], pooled[a][j]) - AUC(y[j], pooled[b][j]))
    print(f'{a:16s} - {b:7s}: {AUC(y, pooled[a]) - AUC(y, pooled[b]):+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
    