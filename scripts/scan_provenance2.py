import os, importlib.util, numpy as np, pandas as pd, pydicom
spec = importlib.util.spec_from_file_location('pp', 'scripts/mri_preprocess_v2.py'); pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
ROOTS = ['data/raw/mri/batch1_combined', 'data/raw/mri/batch2_combined', 'data/raw/mri/batch3_combined', 'data/raw/mri/pilot_download']
ix = pp.index_dicom(ROOTS); print('pipeline indexer: image folders found =', len(ix))
sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()][['RID', 'PTID', 'T_anchor', 'adni_phase', 'label_primary']]
M = sel.merge(coh, on='RID'); M['path'] = M.iid.map(ix); print('selected', len(M), '| located by the pipeline indexer:', int(M.path.notna().sum()))
def root_of(p):
    if not isinstance(p, str): return 'NOT_FOUND'
    for part in p.split(os.sep):
        if part.endswith('_combined') or part == 'pilot_download': return part
    return 'other'
def meta(p):
    n = 0
    for x, _, fs in os.walk(p):
        for f in sorted(fs):
            if f.startswith('.'): continue
            n += 1
            if n > 8: return {}
            try:
                ds = pydicom.dcmread(os.path.join(x, f), stop_before_pixels=True, force=True)
                if 'PatientID' not in ds: continue
                t = getattr(ds, 'MagneticFieldStrength', None); t = float(t) if t not in (None, '') else np.nan; t = t / 10000 if t == t and t > 100 else t
                mf = str(getattr(ds, 'Manufacturer', '')).upper()
                return dict(patient=str(ds.PatientID).strip(), maker='SIEMENS' if 'SIEMENS' in mf else 'PHILIPS' if 'PHILIPS' in mf else 'GE' if 'GE' in mf else (mf[:10] or 'UNK'),
                            tesla=round(t, 1) if t == t else np.nan, date=str(getattr(ds, 'AcquisitionDate', '') or getattr(ds, 'StudyDate', '')), series=str(getattr(ds, 'SeriesDescription', '')))
            except Exception: continue
    return {}
H = pd.DataFrame([meta(p) if isinstance(p, str) else {} for p in M.path]); M = pd.concat([M.reset_index(drop=True), H], axis=1)
for c in ['patient', 'maker', 'tesla', 'date', 'series']:
    if c not in M: M[c] = np.nan
M['root'] = M.path.map(root_of); M['grp'] = np.where(M.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3'); rd = M.patient.notna()
M['pid_match'] = M.patient.astype(str).str.strip() == M.PTID.astype(str).str.strip()
M['days'] = (pd.to_datetime(M.date, format='%Y%m%d', errors='coerce') - pd.to_datetime(M.T_anchor)).dt.days
print(f'header readable: {int(rd.sum())} | PatientID == PTID: {M.pid_match[rd].mean():.3f} | days in [-180, +90]: {M.days[rd].between(-180, 90).mean():.3f}')
print('mismatches (first 10):', M[rd & ~M.pid_match][['RID', 'PTID', 'patient']].head(10).to_dict('records'))
print('\nBy download root and phase group:'); print(M.groupby(['root', 'grp']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
M['seq'] = M.series.astype(str).str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()
print('\nSeries description by root (top 14):'); print(pd.crosstab(M.seq, M.root).assign(t=lambda x: x.sum(1)).sort_values('t', ascending=False).head(14).to_string())
print('\nScanner by root:'); print(pd.crosstab([M.maker.fillna('NA'), M.tesla.fillna(-1)], M.root).to_string())
old = pd.read_csv('results/mri_v2/scan_provenance.csv'); miss = set(old[old.roots == 'NOT_FOUND'].RID); sub = M[M.RID.isin(miss)]
print(f'\nNOT_FOUND in script 1 but located by the pipeline indexer: {int(sub.path.notna().sum())} of {len(sub)}'); print('example paths:', sub.path.dropna().head(5).tolist())
M.drop(columns=['path']).to_csv('results/mri_v2/scan_provenance2.csv', index=False)