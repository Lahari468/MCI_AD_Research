import os, re, glob, numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0); FSL = os.environ['FSLDIR']
img = glob.glob(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'); xml = glob.glob(f'{FSL}/data/atlases/HarvardOxford-Subcortical.xml')
assert img and xml, 'Harvard-Oxford subcortical atlas not found under $FSLDIR/data/atlases'
At = nib.load(img[0]).get_fdata().astype(int); assert At.shape == (91, 109, 91), At.shape
labs = {int(i): n.strip() for i, n in re.findall(r'<label index="(\d+)"[^>]*>([^<]+)</label>', open(xml[0]).read())}
def vox(name):
    k = [i for i, n in labs.items() if n.lower() == name.lower()]; assert k, f'label not found: {name}'; return At == (k[0] + 1)
names = ['Left Hippocampus', 'Right Hippocampus', 'Left Amygdala', 'Right Amygdala', 'Left Lateral Ventricle', 'Right Lateral Ventricle']
ROI = {n: vox(n) for n in names}; WM = vox('Left Cerebral White Matter') | vox('Right Cerebral White Matter')
print('ROI voxel counts (2 mm):', {k: int(v.sum()) for k, v in ROI.items()}, '| WM', int(WM.sum()))
assert 250 < ROI['Left Hippocampus'].sum() < 900, 'hippocampus ROI size implausible; label mapping needs checking'
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
S = coh[coh.RID.isin(dev.rid.unique()) & coh.RID.isin(ok)]; S = S[S.RID.map(lambda r: os.path.exists(f'data/interim/mri_mni2mm_v2/{r}.npy'))].reset_index(drop=True)
y = S.label_primary.astype(int).values; n = len(S); idx = {r: i for i, r in enumerate(S.RID)}; grp = np.where(S.adni_phase == 'ADNI1', 'ADNI1', 'later')
print(f'MRI-usable development participants: {n}, events {y.sum()}')
tens = [np.load(f'data/interim/mri_mni2mm_v2/{r}.npy')[0].astype(np.float32) for r in S.RID]
def roi_feats(x):
    wm = np.median(x[WM]); out = []
    for m in ROI.values(): out += list(np.percentile(x[m], [10, 50, 90]) - wm)
    return out
RF = np.array([roi_feats(x) for x in tens]); T = np.stack([x[::4, ::4, ::4].ravel() for x in tens]); del tens
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(S.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); R = f29.reindex(S.RID)
c29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; X29 = R[c29].astype(float).copy()
for c in c29:
    if c.endswith('SV'): X29[c] = X29[c] / R['ST10CV'] * 1000
X29 = X29.values; ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(S.RID)
mfs = (ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1)).astype(float); X29[mfs == 0] = np.nan
PH = pd.get_dummies(S.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values
hv = ((R.ST29SV + R.ST88SV) / R.ST10CV).values if {'ST29SV', 'ST88SV', 'ST10CV'} <= set(R.columns) else None
if hv is not None:
    u = mfs > 0; print('\nSanity (Spearman with FreeSurfer bilateral hippocampal volume / ICV, usable FS rows):')
    for j, nm in enumerate(names):
        for k, q in enumerate(['p10', 'p50', 'p90']): pass
    for nm, cols in [('hippocampus contrast (median of p50)', [1, 4]), ('amygdala contrast', [7, 10]), ('lateral ventricle contrast', [13, 16])]:
        print(f'  {nm}: rho = {spearmanr(RF[u][:, cols].mean(1), hv[u])[0]:+.3f}')
nb = Xc.shape[1] + X29.shape[1] + 1
def make(kind):
    ph = '+ph' in kind; blocks = [Xc, X29, mfs[:, None]] + ([PH] if ph else []); base = np.hstack(blocks); b = base.shape[1]
    steps = [('base', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(b)))]; X = base
    if 'ROI' in kind: X = np.hstack([base, RF]); steps.append(('roi', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), list(range(b, b + RF.shape[1]))))
    if 'PC10' in kind: X = np.hstack([base, T]); steps.append(('pc', Pipeline([('p', PCA(10, svd_solver='randomized', random_state=0)), ('s', StandardScaler())]), list(range(b, b + T.shape[1]))))
    pipe = Pipeline([('pre', ColumnTransformer(steps)), ('lr', LogisticRegression(max_iter=3000))])
    return X, GridSearchCV(pipe, {'lr__C': np.logspace(-3, 1, 7)}, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
M = ['A', 'A+ph', 'A+ROI', 'A+ph+ROI', 'A+PC10', 'A+ph+PC10']; reps = sorted(dev.repeat.unique()); res = {m: {r: np.full(n, np.nan) for r in reps} for m in M}
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        v = np.array(sorted(set(g.rid.values) & set(S.RID))); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(S.RID.values, v)])
for m in M:
    X, gs = make(m); print('running', m, flush=True)
    for rep in reps:
        for val, trn in folds(rep): res[m][rep][val] = gs.fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
pooled = {m: np.mean([res[m][r] for r in reps], 0) for m in M}; print('\nAUROC (pooled over repeats):', {m: round(float(AUC(y, pooled[m])), 4) for m in M})
def dci(a, b, mask=None):
    ids = np.arange(n) if mask is None else np.where(mask)[0]; pa, pb, yy = pooled[a][ids], pooled[b][ids], y[ids]; d = []
    for _ in range(1500):
        j = rng.integers(0, len(ids), len(ids))
        if 0 < yy[j].sum() < len(j): d.append(AUC(yy[j], pa[j]) - AUC(yy[j], pb[j]))
    return float(AUC(yy, pa) - AUC(yy, pb)), np.percentile(d, [2.5, 97.5]), np.percentile(d, [0.5, 99.5])
print('\n=== Paired increments: with image features minus without (identical folds, fully nested preprocessing) ===')
for ref, new in [('A', 'A+ROI'), ('A+ph', 'A+ph+ROI'), ('A', 'A+PC10'), ('A+ph', 'A+ph+PC10')]:
    dl, c95, c99 = dci(new, ref); per = [float(AUC(y, res[new][r]) - AUC(y, res[ref][r])) for r in reps]; g1 = dci(new, ref, grp == 'ADNI1')[0]; g2 = dci(new, ref, grp == 'later')[0]
    print(f'{new:11s} - {ref:5s}: {dl:+.4f} 95% [{c95[0]:+.4f}, {c95[1]:+.4f}] 99% [{c99[0]:+.4f}, {c99[1]:+.4f}] | per-repeat {[round(p, 4) for p in per]} | ADNI1 {g1:+.4f} later {g2:+.4f} | criterion met: {bool(dl >= 0.010 and c99[0] > 0 and sum(p > 0 for p in per) >= 2 and g1 > 0 and g2 > 0)}')