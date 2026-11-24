import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t = pd.read_csv('results/paper/T3_paired_differences.csv')[['comparison', 'dAUROC', 'lo', 'hi']]
extra = pd.DataFrame([('† ROI contrast features (nested)', 0.0002, -0.0050, 0.0053), ('† 10 image components (nested)', -0.0040, -0.0090, 0.0002), ('† 225 extra FreeSurfer measures', 0.0038, -0.0121, 0.0190)], columns=t.columns)
t = pd.concat([t, extra]).iloc[::-1].reset_index(drop=True); fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(t.dAUROC, range(len(t)), xerr=[t.dAUROC - t.lo, t.hi - t.dAUROC], fmt='o', capsize=3, ms=4); ax.axvline(0, color='k', lw=0.8)
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.comparison, fontsize=7); ax.set_xlabel('Paired \u0394AUROC (95% participant-bootstrap CI), development'); plt.tight_layout(); plt.savefig('results/paper/F_forest.png', dpi=150)