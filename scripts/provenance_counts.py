import re, pandas as pd, numpy as np
norm = lambda s: re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()
rec = pd.read_csv('results/mri_v2/recovered_193_manifest_comparison.csv'); print('recovered rows:', len(rec), '| unique image IDs:', rec.image_id.nunique())
for c in ['pid_match', 'patient_match', 'date_in_window']:
    if c in rec: print(c, rec[c].astype(str).value_counts().to_dict())
if 'header_errors' in rec: he = rec.header_errors.astype(str).str.strip(); print('non-empty header_errors:', int((~he.isin(['', '0', '0.0', '[]', 'nan', 'None'])).sum()))
if {'dicom_files', 'readable_headers'} <= set(rec.columns):
    a, b = pd.to_numeric(rec.dicom_files, errors='coerce'), pd.to_numeric(rec.readable_headers, errors='coerce')
    print('series with all headers readable:', int((a == b).sum()), 'of', len(rec), '| total DICOM files:', int(a.sum()))
man = pd.read_csv('data/raw/mri_meta/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv'); man['image_id'] = man.image_id.astype(str).str.replace(r'\D', '', regex=True); md = dict(zip(man.image_id, man.image_description))
iid = rec.iid.astype(str).str.replace(r'\D', '', regex=True); hd = rec.series_descriptions.astype(str)
exact = [norm(md.get(i, '#none#')) in [norm(x) for x in re.split(r"[;|\[\]']+", h) if x.strip()] for i, h in zip(iid, hd)]
sub = [norm(md.get(i, '#none#')) in norm(h) for i, h in zip(iid, hd)]
print('recovered: manifest description == header description (token match):', int(sum(exact)), '| substring match:', int(sum(sub)), 'of', len(rec))
bad = [(i, md.get(i), h) for i, h, e in zip(iid, hd, exact) if not e][:8]; print('examples not matched:', bad)
p4 = pd.read_csv('results/mri_v2/scan_provenance4.csv'); o = p4[p4.on_disk & p4.series.notna()]
print('earlier 520: manifest description == header series:', int((o.seq == o.series.map(norm)).sum()), 'of', len(o))
g = pd.read_csv('results/mri_v2/recovery_audit/dicom_geometry_713.csv'); print('\ngeometry file:', g.shape); print(list(g.columns))
idc = [c for c in g.columns if 'image' in c.lower() and 'id' in c.lower()]; print('image-id column:', idc, '| unique:', g[idc[0]].nunique() if idc else 'NA')
for c in g.columns:
    if g[c].nunique(dropna=False) <= 12: print('\n', c, g[c].astype(str).value_counts(dropna=False).to_dict())