import os, sys, pandas as pd
roots = sys.argv[1:]
ix = {}
for r in roots:
    for dp, dn, _ in os.walk(r, followlinks=True):
        for d in dn:
            if d[0] == 'I' and d[1:].isdigit(): ix[d[1:]] = dp
coh = pd.read_csv('data/processed/cohort_v2b.csv')
elig = coh[coh.label_primary.notna()][['RID', 'label_primary', 'adni_phase']]
sel = pd.read_csv('data/processed/mri_selected.csv')
sel['image_id'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
sel = elig.merge(sel, on='RID', how='left')
sel['on_disk'] = sel.image_id.map(lambda i: i in ix)
has = sel.image_id.notna()
print('eligible:', len(sel), '| selected scan exists:', has.sum(), '| on disk:', sel.on_disk.sum(),
      '| to download:', (has & ~sel.on_disk).sum(), '| no scan in window:', (~has).sum())
print(sel.groupby('adni_phase').agg(n=('RID', 'size'), have_scan=('image_id', lambda s: s.notna().sum()), on_disk=('on_disk', 'sum')))
print('\nlabel rate by on-disk status (size, positives):')
print(sel[has].groupby('on_disk').label_primary.agg(['size', 'sum']))
sel[has][['RID', 'image_id']].to_csv('data/processed/mri_selected_eligible.csv', index=False)
sel[has & ~sel.on_disk][['RID', 'image_id']].to_csv('data/processed/mri_selected_eligible_missing.csv', index=False)