import pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as im
d = pd.read_csv('results/mri/preproc_log.csv')
d = d[d.status.isin(['OK', 'REVIEW', 'FAIL'])].drop_duplicates('rid', keep='last')
coh = pd.read_csv('data/processed/cohort_v2b.csv')[['RID', 'adni_phase', 'label_primary']]
sel = pd.read_csv('06_audit_outputs/selected_MRI_records_90d.csv')[['RID', 'image_description']].drop_duplicates('RID')
d = d.merge(coh, left_on='rid', right_on='RID').merge(sel, on='RID', how='left')
d['low_ncc'] = d.ncc < 0.4
print(d.groupby('adni_phase').agg(n=('rid', 'size'), low_ncc=('low_ncc', 'mean'), ncc_med=('ncc', 'median'), dice_med=('dice', 'median')).round(2))
print(d.groupby('image_description').agg(n=('rid', 'size'), low_ncc=('low_ncc', 'mean'), ncc_med=('ncc', 'median')).query('n>=8').sort_values('n', ascending=False).head(12).round(2))
print(d.groupby('low_ncc').label_primary.agg(['size', 'mean']).round(3))
w = d[(d.ncc < 0.3) & (d.dice > 0.85)].sample(12, random_state=0)
fig, ax = plt.subplots(4, 3, figsize=(15, 10))
for a_, r in zip(ax.ravel(), w.itertuples()):
    a_.imshow(im.imread(f'results/mri/qc/{r.rid}.png')); a_.axis('off')
plt.tight_layout(); plt.savefig('results/mri/qc_lowncc_goodDice.png', dpi=80)