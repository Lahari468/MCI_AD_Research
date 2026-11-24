import os, re, glob, numpy as np, pandas as pd, nibabel as nib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0); FSL = os.environ['FSLDIR']; os.makedirs('results/mri_fallback', exist_ok=True)
img = glob.glob(f'{FSL}/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'); xml = glob.glob(f'{FSL}/data/atlases/HarvardOxford-Subcortical.xml'); assert img and xml
At = nib.load(img[0]).get_fdata().astype(int); labs = {int(i): n.strip() for i, n in re.findall(r'<label index="(\d+)"[^>]*>([^<]+)</label>', open(xml[0]).read())}
def vox(name):
    k = [i for i, n in labs.items() if n.lower() == name.lower()]; assert k, name; return At == (k[0] + 1)
ROI = {n: vox(n) for n in ['Left Hippocampus', 'Right Hippocampus', 'Left Amygdala', 'Right Amygdala', 'Left Lateral Ventricle', 'Right Lateral Ventricle']}; WM = vox('Left Cerebral White Matter') | vox('Right Cerebral White Matter')
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].reset_index(drop=True)
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); dev = sp[sp.split != 'test']; test_ids = set(sp[sp.split == 'test'].rid)
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
S = coh[coh.RID.isin(dev.rid.unique()) & coh.RID.isin(ok)]; S = S[S.RID.map(lambda r: os.path.exists(f'data/interim/mri_mni2mm_v2/{r}.npy'))].reset_index(drop=True)
assert not (set(S.RID) & test_ids), 'locked-test participant present'
y = S.label_primary.astype(int).values; n = len(S); idx = {r: i for i, r in enumerate(S.RID)}; grp = np.where(S.adni_phase == 'ADNI1', 'ADNI1', 'later'); print(f'MRI-usable development participants: {n}, events {y.sum()}')
def roi_feats(x):
    wm = np.median(x[WM]); out = []
    for m in ROI.values(): out += list(np.percentile(x[m], [10, 50, 90]) - wm)
    return out
RF = np.array([roi_feats(np.load(f'data/interim/mri_mni2mm_v2/{r}.npy')[0].astype(np.float32)) for r in S.RID])
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
Xc = cl.reindex(S.RID)[['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']].astype(float).values
f29 = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID'); R = f29.reindex(S.RID)
c29 = [c for c in f29.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]; X29 = R[c29].astype(float).copy()
for c in c29:
    if c.endswith('SV'): X29[c] = X29[c] / R['ST10CV'] * 1000
X29 = X29.values; ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID').reindex(S.RID)
mfs = (ms.freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(X29).all(1)).astype(float); X29[mfs == 0] = np.nan
PH = pd.get_dummies(S.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0).astype(float).values
g = pd.read_csv('results/mri_v2/recovery_audit/dicom_geometry_713.csv'); low = {c: c.lower() for c in g.columns}
cr = [c for c in g.columns if low[c] == 'rid']; ci = [c for c in g.columns if 'image' in low[c] and 'id' in low[c]]
cm = [c for c in g.columns if re.search('manufact|maker', low[c])]; cf = [c for c in g.columns if re.search('tesla|field|strength', low[c])]
print('geometry columns used:', dict(rid=cr, image=ci, manufacturer=cm, field=cf)); assert (cr or ci) and cm and cf, 'cannot detect geometry columns; send me the header'
if cr: g['_k'] = g[cr[0]]; S['_k'] = S.RID
else:
    sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
    g['_k'] = g[ci[0]].astype(str).str.replace(r'\D', '', regex=True); S = S.merge(sel[['RID', 'iid']], on='RID', how='left'); S['_k'] = S.iid
gg = g.drop_duplicates('_k').set_index('_k').reindex(S['_k'])
def tb(v):
    try: v = float(v)
    except Exception: return 'UNK'
    if v != v: return 'UNK'
    if v > 1000: v = v / 10000
    return '3T' if v >= 2.4 else '1.5T'
mk = gg[cm[0]].astype(str).str.upper().map(lambda s: 'SIEMENS' if 'SIEMENS' in s else 'PHILIPS' if 'PHILIPS' in s else 'GE' if 'GE' in s else 'OTHER').values; fb = gg[cf[0]].map(tb).values
print('protocol groups:', pd.Series([f'{a} {b}' for a, b in zip(mk, fb)]).value_counts().to_dict())
PROT = pd.get_dummies(pd.Series(mk).astype(str) + '_' + pd.Series(fb).astype(str)).astype(float).values
B = {'cl': Xc, 'fs': np.hstack([X29, mfs[:, None]]), 'roi': RF, 'ph': PH, 'prot': PROT}
M = {'C': ['cl'], 'C+ROI': ['cl', 'roi'], 'C+ph': ['cl', 'ph'], 'C+ph+ROI': ['cl', 'ph', 'roi'], 'C+prot': ['cl', 'prot'], 'C+prot+ROI': ['cl', 'prot', 'roi'], 'F': ['cl', 'fs'], 'F+ROI': ['cl', 'fs', 'roi']}
CON = [('C+ROI', 'C'), ('C+ph+ROI', 'C+ph'), ('C+prot+ROI', 'C+prot'), ('F', 'C'), ('F', 'C+ROI'), ('F+ROI', 'F')]; CRIT = {0, 1, 2}
def make(parts):
    arrs = [B[p] for p in parts]; X = np.hstack(arrs); steps = []; s = 0
    for p, a in zip(parts, arrs):
        cols = list(range(s, s + a.shape[1])); s += a.shape[1]
        t = Pipeline([('s', StandardScaler())]) if p in ('ph', 'prot') else Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]); steps.append((p, t, cols))
    pipe = Pipeline([('pre', ColumnTransformer(steps)), ('lr', LogisticRegression(max_iter=3000))])
    return X, GridSearchCV(pipe, {'lr__C': np.logspace(-3, 1, 7)}, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        gq = d[d.fold == f]
        if gq.split.nunique() > 1: gq = gq[gq.split.str.lower().str.startswith('val')]
        v = np.array(sorted(set(gq.rid.values) & set(S.RID))); yield np.array([idx[r] for r in v]), np.array([idx[r] for r in np.setdiff1d(S.RID.values, v)])
reps = sorted(dev.repeat.unique()); res = {m: {r: np.full(n, np.nan) for r in reps} for m in M}
for m, parts in M.items():
    X, gs = make(parts); print('running', m, flush=True)
    for rep in reps:
        for val, trn in folds(rep): res[m][rep][val] = gs.fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
pooled = {m: np.mean([res[m][r] for r in reps], 0) for m in M}
pd.DataFrame([dict(rid=S.RID[i], repeat=r, model=m, prob=res[m][r][i]) for m in M for r in reps for i in range(n)]).to_csv('results/mri_fallback/oof.csv', index=False)
print('\nAUROC pooled | per-repeat mean +- SD | ADNI1 | later')
for m in M:
    pr = [AUC(y, res[m][r]) for r in reps]; print(f'{m:12s} {AUC(y, pooled[m]):.4f} | {np.mean(pr):.4f} +- {np.std(pr):.4f} | {AUC(y[grp == "ADNI1"], pooled[m][grp == "ADNI1"]):.4f} | {AUC(y[grp == "later"], pooled[m][grp == "later"]):.4f}')
def dci(a, b, mask=None):
    ids = np.arange(n) if mask is None else np.where(mask)[0]; pa, pb, yy = pooled[a][ids], pooled[b][ids], y[ids]; d = []
    for _ in range(2000):
        j = rng.integers(0, len(ids), len(ids))
        if 0 < yy[j].sum() < len(j): d.append(AUC(yy[j], pa[j]) - AUC(yy[j], pb[j]))
    return float(AUC(yy, pa) - AUC(yy, pb)), np.percentile(d, [2.5, 97.5]), np.percentile(d, [0.5, 99.5])
print('\nPaired contrasts (first minus second); criterion evaluated for the first three only')
for k, (a, b) in enumerate(CON):
    dl, c95, c99 = dci(a, b); per = [float(AUC(y, res[a][r]) - AUC(y, res[b][r])) for r in reps]; g1, g2 = dci(a, b, grp == 'ADNI1')[0], dci(a, b, grp == 'later')[0]
    cr_txt = f' | criterion met: {bool(dl >= 0.010 and c99[0] > 0 and sum(p > 0 for p in per) >= 2 and g1 > 0 and g2 > 0)}' if k in CRIT else ''
    print(f'{a:11s} - {b:9s}: {dl:+.4f} 95% [{c95[0]:+.4f}, {c95[1]:+.4f}] 99% [{c99[0]:+.4f}, {c99[1]:+.4f}] | per-repeat {[round(p, 4) for p in per]} | ADNI1 {g1:+.4f} later {g2:+.4f}{cr_txt}')