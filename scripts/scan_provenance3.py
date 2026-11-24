import numpy as np, pandas as pd
sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['image_id'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
rec = pd.read_csv('06_audit_outputs/selected_MRI_records_90d.csv'); rec['image_id'] = rec.image_id.astype(str).str.replace(r'\D', '', regex=True)
dc = 'image_date_parsed' if 'image_date_parsed' in rec else 'image_date'; rec['rdate'] = pd.to_datetime(rec[dc], errors='coerce')
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()][['RID', 'PTID', 'T_anchor', 'adni_phase', 'label_primary']]
prov = pd.read_csv('results/mri_v2/scan_provenance2.csv')[['RID', 'root', 'patient', 'series']]
M = (sel.merge(coh, on='RID').merge(rec[['image_id', 'subject_id', 'rdate', 'image_description']].drop_duplicates('image_id'), on='image_id', how='left').merge(prov, on='RID', how='left'))
M['on_disk'] = M.root.ne('NOT_FOUND'); M['round'] = np.where(M.root == 'batch3_combined', 'final round (batch3)', 'earlier rounds'); M['grp'] = np.where(M.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3')
M['days'] = (M.rdate - pd.to_datetime(M.T_anchor)).dt.days; M['subj_ok'] = M.subject_id.astype(str).str.strip() == M.PTID.astype(str).str.strip()
print('selected scans:', len(M), '| found in metadata:', int(M.subject_id.notna().sum()))
for nm, g in [('all 713', M), ('on disk', M[M.on_disk]), ('deleted DICOMs', M[~M.on_disk])]:
    print(f'{nm:15s} n={len(g)} | subject ID == PTID: {g.subj_ok.mean():.3f} | date inside [-180, +90]: {g.days.between(-180, 90).mean():.3f} | date missing: {int(g.days.isna().sum())}')
print('mismatches (first 10):', M[~M.subj_ok][['RID', 'PTID', 'subject_id']].head(10).to_dict('records'))
M['seq'] = M.image_description.astype(str).str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()
print('\nSeries description (manifest) by download round:'); print(pd.crosstab(M.seq, M['round']).assign(t=lambda x: x.sum(1)).sort_values('t', ascending=False).head(14).to_string())
print('\nProgression by round and phase group:'); print(M.groupby(['round', 'grp']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
ok = M[M.on_disk & M.series.notna()]
if len(ok): print('\nmanifest description matches DICOM header series on disk:', f'{(ok.seq == ok.series.astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()).mean():.3f}')
import os; os.makedirs('results/coverage', exist_ok=True)
open('results/coverage/redownload_deleted.txt', 'w').write('\n'.join('I' + i for i in M[~M.on_disk].image_id)); print('\nre-download list written:', int((~M.on_disk).sum()), 'IDs')
M.drop(columns=['root']).to_csv('results/mri_v2/scan_provenance3.csv', index=False)