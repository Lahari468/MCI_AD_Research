import pandas as pd
d = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
d['st'] = d.status.str[:6]
coh = pd.read_csv('data/processed/cohort_v2b.csv')[['RID', 'adni_phase', 'label_primary']].dropna(subset=['label_primary'])
m = coh.merge(d[['rid', 'st']], left_on='RID', right_on='rid', how='left'); m['st'] = m.st.fillna('NO_SCAN')
print(m.groupby('st').label_primary.agg(['size', 'sum', 'mean']).round(3)); print(pd.crosstab(m.adni_phase, m.st))
print(d[d.status.str.startswith('ERROR')][['rid', 'status']].to_string())