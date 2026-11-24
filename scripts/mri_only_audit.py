import re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
print('=== 1. pid_match vs patient_match (recovered 193) ===')
rec = pd.read_csv('results/mri_v2/recovered_193_manifest_comparison.csv')
print('nulls in earlier-merge columns:', rec[['patient', 'maker', 'tesla', 'date', 'series', 'root']].isna().sum().to_dict())
exp = rec.PTID_expected.astype(str).str.strip(); hdr = rec.patient_ids.astype(str).str.strip("[]'\" ")
print('header patient_ids == PTID_expected (direct):', int((hdr == exp).sum()), 'of', len(rec), '| PTID == PTID_expected:', int((rec.PTID.astype(str).str.strip() == exp).sum()))
print('examples:', rec[['patient', 'patient_ids', 'PTID_expected']].head(3).to_dict('records'))
print('\n=== 2. split file structure ===')
sp = pd.read_csv('data/processed/splits/splits_v1.csv'); print(sp.split.value_counts().to_dict(), '| repeats:', sorted(sp.repeat.unique()), '| folds:', sorted(sp.fold.unique()))
dev, test = sp[sp.split != 'test'], set(sp[sp.split == 'test'].rid); print('dev rids:', dev.rid.nunique(), '| test rids:', len(test), '| overlap:', len(set(dev.rid) & test))
log = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); ok = set(log[log.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)
for r in sorted(dev.repeat.unique()):
    d = dev[dev.repeat == r]; m = d[d.rid.isin(ok)]
    print(f'repeat {r}: rows {len(d)}, unique rids {d.rid.nunique()} (each participant once: {len(d) == d.rid.nunique()}) | folds {d.groupby("fold").size().to_dict()} | MRI-usable per fold {m.groupby("fold").size().to_dict()}')
print('\n=== 3. MRI-only predictions: structure and recomputed AUROCs ===')
oof = pd.read_csv('results/mri_only/oof.csv'); coh = pd.read_csv('data/processed/cohort_v2b.csv').set_index('RID'); y = coh.label_primary
print('rows', len(oof), '| participants', oof.rid.nunique(), '| models', oof.model.nunique(), '| repeats', oof['repeat'].nunique(), '| NaN probs', int(oof.prob.isna().sum()), '| test participants present:', len(set(oof.rid) & test))
for m, g in oof.groupby('model'):
    p = g.groupby('rid').prob.mean(); per = [AUC(y.loc[gg.rid].astype(int), gg.prob) for _, gg in g.groupby('repeat')]
    print(f'{m:16s} pooled {AUC(y.loc[p.index].astype(int), p):.4f} | per-repeat {np.mean(per):.4f} +- {np.std(per):.4f}')
print('\n=== 4. field strength raw values (geometry file) ===')
gm = pd.read_csv('results/mri_v2/recovery_audit/dicom_geometry_713.csv'); print(list(gm.columns))
for c in [c for c in gm.columns if re.search('tesla|field|strength', c, re.I)]: print(c, gm[c].astype(str).value_counts().head(15).to_dict())