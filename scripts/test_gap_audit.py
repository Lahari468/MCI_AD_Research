import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
rng = np.random.default_rng(0)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); Y = coh[coh.label_primary.notna()].set_index('RID').label_primary.astype(int)
sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid'); test = set(sp[sp.split == 'test'].index)
def load(run, cond='full'):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == cond]; return d.groupby('rid').prob.mean()
def boot(y, p, n=2000):
    out = []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y): out.append(AUC(y[i], p[i]))
    return np.array(out)
PAIRS = {'LR clinical + fs29 (primary)': ('lr_full_fs29', 'final_lr_full_fs29'), 'LR clinical only': ('lr_full_clinonly', 'final_lr_full_clinonly'),
         'LR no-circular + fs29': ('lr_nocirc_fs29', 'final_lr_nocirc_fs29'), 'Attention': ('attn_full_fs29', 'final_attn_full_fs29'),
         'Concatenation': ('concat_full_fs29', 'final_concat_full_fs29'), 'Attention + MRI (dev: one repeat)': ('mri_fs29_drop', 'final_mri_fs29_drop')}
rows = []
for k, (a, b) in PAIRS.items():
    d, t = load(a), load(b); yd, yt = Y.loc[d.index].values, Y.loc[t.index].values; diff = boot(yt, t.values) - boot(yd, d.values)
    rows.append(dict(model=k, dev=AUC(yd, d.values), test=AUC(yt, t.values), test_minus_dev=AUC(yt, t.values) - AUC(yd, d.values), lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
print('=== Dev vs locked-test AUROC per model (independent participant bootstraps for the difference) ==='); print(pd.DataFrame(rows).round(3).to_string(index=False))
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID'); fsd = pd.read_csv('data/processed/features/freesurfer29_raw_v1.csv').set_index('RID')
hv = (fsd.ST29SV + fsd.ST88SV) / fsd.ST10CV
F = {'MMSE (lower = worse)': -cl.MMSCORE, 'CDR-SB': cl.CDRSB, 'FAQ': cl.FAQTOTAL, 'ADAS-Cog 13': cl.TOTAL13, 'Hippocampal volume / ICV (lower = worse)': -hv, 'Age': cl.age}
dids = [r for r in Y.index if r not in test]; tids = [r for r in Y.index if r in test]; rows = []
for k, s in F.items():
    sd, st = s.reindex(dids), s.reindex(tids); md, mt = sd.notna().values, st.notna().values; yd, yt = Y.loc[dids].values[md], Y.loc[tids].values[mt]
    diff = boot(yt, st.values[mt], 1000) - boot(yd, sd.values[md], 1000)
    rows.append(dict(single_feature=k, dev_auroc=AUC(yd, sd.values[md]), test_auroc=AUC(yt, st.values[mt]), diff=AUC(yt, st.values[mt]) - AUC(yd, sd.values[md]), lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
print('\n=== Single-feature AUROC (directions fixed in advance by clinical knowledge; no fitting; descriptive, post hoc) ==='); print(pd.DataFrame(rows).round(3).to_string(index=False))