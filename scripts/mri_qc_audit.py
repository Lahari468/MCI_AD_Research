import os, sys, re
import numpy as np, pandas as pd, nibabel as nib, pydicom
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score as AUC
EXTRA = [int(a) for a in sys.argv[1:]]
ROOTS = ['data/raw/mri/batch1_combined', 'data/raw/mri/batch2_combined', 'data/raw/mri/batch3_combined', 'data/raw/mri/pilot_download']
FSL = os.environ['FSLDIR']; os.makedirs('results/mri_v2', exist_ok=True)
T = nib.load(f'{FSL}/data/standard/MNI152_T1_2mm_brain.nii.gz').get_fdata().astype(np.float32)
M = nib.load(f'{FSL}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz').get_fdata() > 0
Tm = T[M]; Tm = (Tm - Tm.mean()) / Tm.std()
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
log['st'] = log.status.astype(str).str[:6].str.strip(': ')
print('=== Observed QC metrics by status (recovers the implemented thresholds) ===')
print(log.groupby('st')[['ncc', 'scale', 'angle']].agg(['min', 'max']).round(2).to_string())
ix = {}
for r in ROOTS:
    for dp, dn, _ in os.walk(r, followlinks=True):
        for d in dn:
            if d[0] == 'I' and d[1:].isdigit(): ix[d[1:]] = os.path.join(dp, d)
def hdr(p):
    try:
        f = sorted(os.listdir(p))[0]; d = pydicom.dcmread(os.path.join(p, f), stop_before_pixels=True, force=True)
        mf = str(getattr(d, 'Manufacturer', 'NA')).upper(); fs = float(getattr(d, 'MagneticFieldStrength', np.nan)); fs = fs / 10000 if fs > 100 else fs
        return ('SIEMENS' if 'SIEMENS' in mf else 'PHILIPS' if 'PHILIPS' in mf else 'GE' if 'GE' in mf else mf[:12]), round(fs, 1)
    except Exception: return 'NA', np.nan
sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()][['RID', 'adni_phase', 'label_primary']]
d = log.merge(sel[['RID', 'iid']], left_on='rid', right_on='RID').merge(coh, on='RID')
def path(r):
    for dd in ('data/interim/mri_mni2mm_v2', 'data/interim/mri_mni2mm_v2_qcfail'):
        p = f'{dd}/{r}.npy'
        if os.path.exists(p): return p
d = d[d.rid.map(lambda r: path(r) is not None)].reset_index(drop=True)
H = [hdr(ix[i]) if i in ix else ('NA', np.nan) for i in d.iid]; d['maker'] = [h[0] for h in H]; d['tesla'] = [h[1] for h in H]
load = lambda r: np.load(path(r))[0].astype(np.float32)
def feats(x):
    xm, xf = x[M], x[::-1][M]; c = lambda a: float(np.corrcoef(a, Tm)[0, 1])
    return dict(cover=float((xm != 0).mean()), mean_in=float(xm.mean()), sd_in=float(xm.std()), flip_margin=c(xm) - c(xf))
d = pd.concat([d, pd.DataFrame([feats(load(r)) for r in d.rid])], axis=1)
print('\n=== By scanner and field strength ===')
print(d.groupby(['maker', 'tesla'], dropna=False).agg(n=('rid', 'size'), fail=('st', lambda s: int((s == 'FAIL').sum())), review=('st', lambda s: int((s == 'REVIEW').sum())),
      ncc=('ncc', 'median'), cover=('cover', 'median'), flip=('flip_margin', 'median'), progress=('label_primary', 'mean')).round(3).to_string())
print('\n', pd.crosstab(d.maker, d.adni_phase).to_string())
print('\nflip_margin quantiles (0, 1%, 5%, 50%):', d.flip_margin.quantile([0, .01, .05, .5]).round(4).tolist(), '| negative:', int((d.flip_margin < 0).sum()))
rz = lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-9)
flag = (d.flip_margin < 0) | (rz(d.cover).abs() > 4) | (rz(d.mean_in).abs() > 4) | (rz(d.sd_in).abs() > 4) | (d.st == 'FAIL') | d.rid.isin(EXTRA)
aud = d[flag]; print('\naudit list:', len(aud), 'scans'); print(aud[['rid', 'st', 'maker', 'tesla', 'ncc', 'scale', 'angle', 'cover', 'flip_margin', 'adni_phase']].round(3).to_string(index=False))
d.to_csv('results/mri_v2/audit_table.csv', index=False); aud.to_csv('results/mri_v2/audit_list.csv', index=False)
u = d[d.st.isin(['OK', 'REVIEW'])].reset_index(drop=True)
X = PCA(30, random_state=0).fit_transform(np.stack([load(r)[::4, ::4, ::4].ravel() for r in u.rid]))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print('\n=== Can the saved tensors reveal scanner, field strength or phase? (5-fold CV AUROC, diagnostic only) ===')
for nm, t in [('Philips vs other', (u.maker == 'PHILIPS')), ('3T vs 1.5T', (u.tesla >= 2.5)), ('ADNI1 vs later', (u.adni_phase == 'ADNI1'))]:
    t = t.astype(int).values
    print(f'{nm}: AUROC {AUC(t, cross_val_predict(LogisticRegression(max_iter=2000), X, t, cv=cv, method="predict_proba")[:, 1]):.3f}  (positives {t.sum()}/{len(t)})')
def sheet(rids, name):
    for k in range(0, len(rids), 8):
        part = list(rids[k:k + 8]); fig, ax = plt.subplots(len(part), 3, figsize=(7.5, 2.0 * len(part))); ax = np.atleast_2d(ax)
        for a, r in zip(ax, part):
            x = load(r); w = d[d.rid == r].iloc[0]
            for j, (s, m) in enumerate([(x[45], M[45]), (x[:, 54], M[:, 54]), (x[:, :, 45], M[:, :, 45])]):
                a[j].imshow(s.T, cmap='gray', origin='lower'); a[j].contour(m.T.astype(float), levels=[0.5], colors='r', linewidths=0.5); a[j].axis('off')
            a[0].set_title(f'{r} {w.st} {w.maker} {w.tesla}T ncc={w.ncc:.2f} flip={w.flip_margin:+.3f}', fontsize=6, loc='left')
        plt.tight_layout(); plt.savefig(f'results/mri_v2/{name}_{k // 8 + 1}.png', dpi=70); plt.close()
rng = np.random.default_rng(0)
for mk in ['SIEMENS', 'GE', 'PHILIPS']:
    g = u[u.maker == mk].rid.values
    if len(g): sheet(rng.choice(g, min(8, len(g)), replace=False), f'audit_{mk}')
sheet(aud.rid.values[:24], 'audit_flagged'); sheet(d.sort_values('flip_margin').rid.values[:8], 'audit_lowest_flip')
print('\nsheets written to results/mri_v2/audit_*.png')