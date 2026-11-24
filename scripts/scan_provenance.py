import os, re, numpy as np, pandas as pd, pydicom
ROOTS = ['data/raw/mri/batch1_combined', 'data/raw/mri/batch2_combined', 'data/raw/mri/batch3_combined', 'data/raw/mri/pilot_download']
sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()][['RID', 'PTID', 'T_anchor', 'adni_phase', 'label_primary']]
d = sel.merge(coh, on='RID'); want = set(d.iid); print('selected scans:', len(d))
found, stats, fpat = {}, {}, re.compile(r'_I(\d+)(?:\.\w+)?$')
for r in ROOTS:
    nm = os.path.basename(r); nd = nf = 0
    for dp, dn, fs in os.walk(r, followlinks=True):
        m = re.fullmatch(r'I(\d+)', os.path.basename(dp))
        if m:
            nd += 1
            if m.group(1) in want: found.setdefault(m.group(1), []).append((nm, dp))
            continue
        for f in fs:
            if f.startswith('.'): continue
            nf += 1; m2 = fpat.search(f)
            if m2 and m2.group(1) in want and not any(dp == p for _, p in found.get(m2.group(1), [])): found.setdefault(m2.group(1), []).append((nm, dp))
    stats[nm] = (nd, nf)
print('(I-folders, loose files outside I-folders) per root:', stats)
def meta(dp):
    tried = 0
    for x, _, fs in os.walk(dp):
        for f in sorted(fs):
            if f.startswith('.'): continue
            tried += 1
            if tried > 6: return {}
            try:
                ds = pydicom.dcmread(os.path.join(x, f), stop_before_pixels=True, force=True)
                if 'PatientID' not in ds: continue
                t = getattr(ds, 'MagneticFieldStrength', None); t = float(t) if t not in (None, '') else np.nan
                t = t / 10000 if t == t and t > 100 else t; mf = str(getattr(ds, 'Manufacturer', '')).upper()
                dt = str(getattr(ds, 'AcquisitionDate', '') or getattr(ds, 'StudyDate', '') or getattr(ds, 'SeriesDate', ''))
                return dict(patient=str(ds.PatientID).strip(), maker='SIEMENS' if 'SIEMENS' in mf else 'PHILIPS' if 'PHILIPS' in mf else 'GE' if 'GE' in mf else (mf[:10] or 'UNK'),
                            tesla=round(t, 1) if t == t else np.nan, date=dt, series=str(getattr(ds, 'SeriesDescription', '')))
            except Exception: continue
    return {}
rows = []
for r in d.itertuples():
    locs = found.get(r.iid, []); row = dict(RID=r.RID, iid=r.iid, n_locations=len(locs), roots='|'.join(sorted({a for a, _ in locs})) or 'NOT_FOUND', path=locs[0][1] if locs else '')
    if locs: row.update(meta(locs[0][1]))
    rows.append(row)
M = pd.DataFrame(rows)
for c in ['patient', 'maker', 'tesla', 'date', 'series']:
    if c not in M: M[c] = np.nan
M = M.merge(d[['RID', 'PTID', 'T_anchor', 'adni_phase', 'label_primary']], on='RID'); M['grp'] = np.where(M.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3')
M['pid_match'] = M.patient.astype(str).str.strip() == M.PTID.astype(str).str.strip()
M['days'] = (pd.to_datetime(M.date, format='%Y%m%d', errors='coerce') - pd.to_datetime(M.T_anchor)).dt.days
rd = M.patient.notna()
print(f'\nlocated: {(M.roots != "NOT_FOUND").sum()} | header readable: {rd.sum()} | PatientID == PTID among readable: {M.pid_match[rd].mean():.3f}')
print('scans with days-from-index inside [-180, +90]:', f'{M.days[rd].between(-180, 90).mean():.3f}', '| outside:', int((~M.days[rd].between(-180, 90)).sum()))
print('mismatching PatientID (first 10):', M[rd & ~M.pid_match][['RID', 'PTID', 'patient']].head(10).to_dict('records'))
print('\nBy root:'); print(M.groupby(['roots', 'grp']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
print('\nBy scanner and field strength:'); print(M.groupby(['maker', 'tesla'], dropna=False).agg(n=('RID', 'size')).to_string())
print('\nProgression within phase group by scanner:'); print(M.groupby(['grp', 'maker'], dropna=False).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
print('\nexample paths:', M.groupby('roots').path.first().to_dict())
M.drop(columns=['path']).to_csv('results/mri_v2/scan_provenance.csv', index=False)