import glob, os, numpy as np, pandas as pd
p = glob.glob('data/raw/mri_meta/T1w_Structural_MRI_Cohort_Manifest*.csv'); assert p, 'copy the manifest into data/raw/mri_meta first'
man = pd.read_csv(p[0], low_memory=False); print('manifest rows:', len(man), '| columns:', list(man.columns)); low = {c: c.lower() for c in man.columns}
def pick(*keys):
    c = [k for k, v in low.items() if all(x in v for x in keys)]; return c[0] if c else None
cid = pick('image', 'id') or pick('image', 'uid'); csub = pick('subject') or pick('ptid'); cdate = pick('acq') or pick('date'); cdesc = pick('desc') or pick('series')
print('detected:', dict(image=cid, subject=csub, date=cdate, description=cdesc)); assert cid and csub and cdate, 'could not detect columns; paste the header to me'
ren = {cid: 'image_id', csub: 'subject_id', cdate: 'rdate_raw'}; keep = [cid, csub, cdate]
if cdesc: ren[cdesc] = 'desc'; keep.append(cdesc)
man = man[keep].rename(columns=ren); man['image_id'] = man.image_id.astype(str).str.replace(r'\D', '', regex=True); man = man.drop_duplicates('image_id')
sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['image_id'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()][['RID', 'PTID', 'T_anchor', 'adni_phase', 'label_primary']]
prov = pd.read_csv('results/mri_v2/scan_provenance2.csv')[['RID', 'root', 'series']]
M = sel.merge(coh, on='RID').merge(man, on='image_id', how='left').merge(prov, on='RID', how='left')
if 'desc' not in M: M['desc'] = np.nan
M['rdate'] = pd.to_datetime(M.rdate_raw, errors='coerce'); M['days'] = (M.rdate - pd.to_datetime(M.T_anchor)).dt.days
M['subj_ok'] = M.subject_id.astype(str).str.strip() == M.PTID.astype(str).str.strip(); M['on_disk'] = M.root.ne('NOT_FOUND')
M['round'] = np.where(M.root == 'batch3_combined', 'final round (batch3)', 'earlier rounds'); M['grp'] = np.where(M.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3')
print('\nselected scans:', len(M), '| found in manifest:', int(M.subject_id.notna().sum()))
for nm, g in [('all', M), ('on disk', M[M.on_disk]), ('deleted DICOMs', M[~M.on_disk])]:
    print(f'{nm:15s} n={len(g)} | subject == PTID: {g.subj_ok.mean():.3f} | date in [-180, +90]: {g.days.between(-180, 90).mean():.3f} | date missing: {int(g.days.isna().sum())}')
print('mismatches (first 10):', M[~M.subj_ok][['RID', 'PTID', 'subject_id']].head(10).to_dict('records'))
norm = lambda s: s.astype(str).str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip(); M['seq'] = norm(M.desc)
print('\nSeries description by round:'); print(pd.crosstab(M.seq, M['round']).assign(t=lambda x: x.sum(1)).sort_values('t', ascending=False).head(14).to_string())
print('\nProgression by round and phase group:'); print(M.groupby(['round', 'grp']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
o = M[M.on_disk & M.series.notna() & M.desc.notna()]
if len(o): print('\nmanifest description == DICOM header series (on-disk scans):', f'{(o.seq == norm(o.series)).mean():.3f}')
os.makedirs('results/coverage', exist_ok=True); open('results/coverage/redownload_deleted.txt', 'w').write('\n'.join('I' + i for i in M[~M.on_disk].image_id))
M.drop(columns=['root']).to_csv('results/mri_v2/scan_provenance4.csv', index=False); print('\nre-download list:', int((~M.on_disk).sum()), 'IDs')