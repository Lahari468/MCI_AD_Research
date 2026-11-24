import pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as im
d = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
print(d.status.str[:6].value_counts().to_string()); print(d[['ncc', 'scale', 'angle', 'secs']].describe().round(2).to_string())
w = d[d.ncc.notna()].sort_values('ncc').head(20)
fig, ax = plt.subplots(5, 4, figsize=(16, 11))
for a_, r in zip(ax.ravel(), w.itertuples()):
    a_.imshow(im.imread(f'results/mri_v2/qc/{r.rid}.png')); a_.axis('off')
plt.tight_layout(); plt.savefig('results/mri_v2/qc_lowest20.png', dpi=80)