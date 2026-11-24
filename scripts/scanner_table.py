import os, numpy as np, pandas as pd, pydicom
ROOTS = ['data/raw/mri/batch1_combined', 'data/raw/mri/batch2_combined', 'data/raw/mri/batch3_combined', 'data/raw/mri/pilot_download']
d = pd.read_csv('results/mri_v2/audit_table.csv').drop(columns=['maker', 'tesla'], errors='ignore'); ix = {}
for r in ROOTS:
    for dp, dn, _ in os.walk(r, followlinks=True):
        for x in dn:
            if x[0] == 'I' and x[1:].isdigit(): ix[x[1:]] = (os.path.join(dp, x), os.path.basename(r))
def hdr(p):
    n = 0
    for dp, _, fs in os.walk(p):
        for f in sorted(fs):
            if f.startswith('.'): continue
            n += 1
            if n > 8: return 'NA', np.nan
            try:
                ds = pydicom.dcmread(os.path.join(dp, f), stop_before_pixels=True); mf = str(getattr(ds, 'Manufacturer', '')).upper()
                if not mf: continue
                t = float(getattr(ds, 'MagneticFieldStrength', np.nan)); t = t / 10000 if t > 100 else t
                return ('SIEMENS' if 'SIEMENS' in mf else 'PHILIPS' if 'PHILIPS' in mf else 'GE' if 'GE' in mf else mf[:10]), round(t, 1)
            except Exception: continue
    return 'NA', np.nan
H = [hdr(ix[str(i)][0]) if str(i) in ix else ('NA', np.nan) for i in d.iid]
d['maker'], d['tesla'] = [h[0] for h in H], [h[1] for h in H]; d['batch'] = [ix[str(i)][1] if str(i) in ix else 'NA' for i in d.iid]
d['grp'] = np.where(d.adni_phase == 'ADNI1', 'ADNI1', 'ADNIGO/2/3'); print('unreadable headers:', int((d.maker == 'NA').sum()), 'of', len(d))
print('\nBy scanner and field strength:'); print(d.groupby(['maker', 'tesla'], dropna=False).agg(n=('rid', 'size'), fail=('st', lambda s: int((s == 'FAIL').sum())), review=('st', lambda s: int((s == 'REVIEW').sum())), ncc=('ncc', 'median')).round(3).to_string())
print('\nProgression within phase group, by scanner:'); print(d.groupby(['grp', 'maker']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
print('\nDownload batch vs progression:'); print(d.groupby(['batch', 'grp']).label_primary.agg(n='size', pos='sum', rate='mean').round(3).to_string())
d.to_csv('results/mri_v2/audit_table_fixed.csv', index=False)