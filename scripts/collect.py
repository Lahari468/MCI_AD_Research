import glob, pandas as pd
rows = []
for p in sorted(glob.glob('results/deep/*/summary.csv')):
    n = p.split('/')[2]
    if n.startswith('smoke'): continue
    s = pd.read_csv(p)
    for _, r in s[s.subset == 'all'].iterrows():
        rows.append(dict(run=n, cond=r.cond, auroc_rep_mean=r.rep_auroc_mean, rep_sd=r.rep_auroc_sd, auroc_pooled=r.auroc, prauc=r.prauc, brier=r.brier))
d = pd.DataFrame(rows).round(3)
print(d[d.cond.isin(['full', 'nofs'])].sort_values(['cond', 'auroc_rep_mean'], ascending=[True, False]).to_string(index=False))
for p in sorted(glob.glob('results/deep/*_full_fs29/summary.csv')):
    s = pd.read_csv(p); s = s[(s.cond == 'full') & s.subset.isin(['ADNI1', 'ADNIGO+2', 'ADNI3'])]
    print(p.split('/')[2], s.set_index('subset').auroc.round(3).to_dict())