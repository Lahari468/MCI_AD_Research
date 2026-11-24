import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID'); fb = pd.read_csv('results/mri_fallback/oof.csv')
rid = np.array(sorted(fb.rid.unique())); y = coh.loc[rid, 'label_primary'].astype(int).values; g1 = (coh.loc[rid, 'adni_phase'] == 'ADNI1').values
pf = fb.groupby(['model', 'rid']).prob.mean().unstack(0).reindex(rid); F, C, CR = pf['F'].values, pf['C'].values, pf['C+ROI'].values
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID'); real = ms.reindex(rid).freesurfer_status.isin(['OBSERVED', 'PARTIAL']).values
def block(sel, name):
    ix = np.where(sel)[0]; yy = y[ix]; Fs, Cs, Rs, rl = F[ix], C[ix], CR[ix], real[ix]
    print(f'\n[{name}] n={len(ix)} events={int(yy.sum())} real FreeSurfer-unavailable={int((~rl).sum())}')
    for p in [0.25, 0.5, 0.75, 1.0]:
        rg = np.random.default_rng(7); out = []
        for _ in range(200):
            un = (~rl) | (rg.random(len(ix)) < p)
            if un.sum() < 20 or yy[un].sum() < 5 or (yy[un] == 0).sum() < 5: continue
            out.append((AUC(yy, np.where(un, Rs, Fs)) - AUC(yy, np.where(un, Cs, Fs)), AUC(yy[un], Rs[un]) - AUC(yy[un], Cs[un]), un.mean()))
        if not out: print(f'  p={p}: no valid masks'); continue
        o = np.array(out); print(f'  p={p:4.2f} masks {len(o):3d} unavailable share {o[:, 2].mean():.2f} | mixture gain {o[:, 0].mean():+.4f} | within-unavailable gain {o[:, 1].mean():+.4f} | excess (mixing) {o[:, 0].mean() - o[:, 1].mean():+.4f}')
block(np.ones(len(rid), bool), 'all'); block(g1, 'ADNI1'); block(~g1, 'later')
