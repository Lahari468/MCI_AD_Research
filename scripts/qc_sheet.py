import pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as im
d = pd.read_csv('results/mri/preproc_log.csv')
print(d.status.value_counts().to_string()); print(d[['ncc', 'dice', 'vol_mm3']].describe().round(2).to_string())
w = d[d.status.isin(['OK', 'REVIEW', 'FAIL'])].sort_values('dice').head(20)
fig, ax = plt.subplots(5, 4, figsize=(16, 11))
for a_, r in zip(ax.ravel(), w.itertuples()):
    a_.imshow(im.imread(f'results/mri/qc/{r.rid}.png')); a_.axis('off')
plt.tight_layout(); plt.savefig('results/mri/qc_worst20.png', dpi=80)