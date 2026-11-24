import os, shutil, sys, platform, subprocess
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
OUT = 'results/paper'; os.makedirs(OUT, exist_ok=True); RNG = np.random.default_rng(0)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); elig = coh[coh.label_primary.notna()].set_index('RID'); Y = elig.label_primary.astype(int)

def save(df, name, scope):
    df.to_csv(f'{OUT}/{name}.csv', index=False)
    open(f'{OUT}/{name}.txt', 'w').write(f'SCOPE: {scope}\n\n' + df.to_string(index=False))
    print(f'\n## {name}  [{scope}]'); print(df.to_string(index=False))
def safe(fn):
    def w():
        try: fn()
        except Exception as e: print(f'\n## {fn.__name__}: SKIPPED ({type(e).__name__}: {e})')
    return w
def load(run, cond='full', rep=None):
    d = pd.read_csv(f'results/deep/{run}/oof.csv'); d = d[d.cond == cond]
    if rep is not None: d = d[d.repeat == rep]
    return d.groupby('rid').prob.mean()
def boots(y, k=2000):
    out = []
    while len(out) < k:
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y): out.append(i)
    return out
def usable_mri():
    l = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last')
    return set(l[l.status.astype(str).str.startswith(('OK', 'REVIEW'))].rid)

@safe
def t1_cohort():
    e = coh.copy(); e['exclusion_reason'] = e.exclusion_reason.fillna('ELIGIBLE')
    save(e.exclusion_reason.value_counts().rename_axis('reason').reset_index(name='n'), 'T1a_cohort_flow_reasons', 'starting cohort N=1199')
    g = elig.groupby('adni_phase').label_primary.agg(n='size', positives='sum').reset_index(); g['rate'] = (g.positives / g.n).round(3)
    save(g, 'T1b_phase_progression', 'eligible cohort N=778')

DEV = ['lr_full_fs29', 'lr_full_clinonly', 'lr_nocirc_fs29', 'concat_full_fs29', 'attn_full_fs29', 'attn_d32', 'attn_d128', 'attn_wd1e2', 'attn_lr1e3',
       'attndrop_full_fs29', 'attndrop_k5_full_fs29', 'concatdrop_full_fs29', 'paperfaith_full_fs29']
@safe
def t2_dev():
    rows = []
    for r in DEV:
        p = f'results/deep/{r}/summary.csv'
        if not os.path.exists(p): print('missing run:', r); continue
        s = pd.read_csv(p); s = s[(s.cond == 'full') & (s.subset == 'all')].iloc[0]
        rows.append(dict(run=r, n=s.n, pos=s.pos, auroc_rep_mean=s.rep_auroc_mean, rep_sd=s.rep_auroc_sd, auroc_pooled=s.auroc, ci_lo=s.auroc_lo, ci_hi=s.auroc_hi, prauc=s.prauc, brier=s.brier, logloss=s.logloss))
    save(pd.DataFrame(rows).round(3), 'T2_dev_cv_models', 'DEVELOPMENT ONLY n=622, 3x5 CV; pooled metrics use repeat-averaged out-of-fold predictions; deep-model Brier is from class-weighted training and not calibrated')

@safe
def t2b_mri():
    rows = []
    for r in ['mri_fs29', 'mri_fs29_drop', 'mri_nofs_drop', 'mri3ch_fs29_drop']:
        p = f'results/deep/{r}/summary.csv'
        if not os.path.exists(p): print('missing run:', r); continue
        s = pd.read_csv(p); s = s[s.subset.isin(['all', 'mri_present', 'mri_absent'])]
        for x in s.itertuples(): rows.append(dict(run=r, cond=x.cond, subset=x.subset, n=x.n, pos=x.pos, auroc=x.auroc, ci_lo=x.auroc_lo, ci_hi=x.auroc_hi, prauc=x.prauc, brier=x.brier))
    save(pd.DataFrame(rows).round(3), 'T2b_mri_models', 'DEVELOPMENT ONLY, one repeat; cond: full=all modalities, nomri=MRI masked at test, nofs=FreeSurfer masked, clin=clinical only')

PAIRS = [
 ('FreeSurfer increment (LR clinical+fs29 minus LR clinical)', 'lr_full_fs29', 'full', 'lr_full_clinonly', 'full', 'all', None),
 ('Attention minus concat', 'attn_full_fs29', 'full', 'concat_full_fs29', 'full', 'all', None),
 ('Attention minus LR', 'attn_full_fs29', 'full', 'lr_full_fs29', 'full', 'all', None),
 ('Paper-scale attention minus LR', 'paperfaith_full_fs29', 'full', 'lr_full_fs29', 'full', 'all', None),
 ('Paper-scale minus reduced attention', 'paperfaith_full_fs29', 'full', 'attn_full_fs29', 'full', 'all', None),
 ('Experts K=5 minus K=1 (with modality dropout)', 'attndrop_k5_full_fs29', 'full', 'attndrop_full_fs29', 'full', 'all', None),
 ('Modality dropout minus none (attention)', 'attndrop_full_fs29', 'full', 'attn_full_fs29', 'full', 'all', None),
 ('No-circular minus full clinical (LR+fs29)', 'lr_nocirc_fs29', 'full', 'lr_full_fs29', 'full', 'all', None),
 ('Attention with FS masked minus LR clinical-only', 'attndrop_full_fs29', 'nofs', 'lr_full_clinonly', 'full', 'all', None),
 ('MRI increment 1ch, dropout (full minus MRI masked)', 'mri_fs29_drop', 'full', 'mri_fs29_drop', 'nomri', 'mri', 0),
 ('MRI increment 1ch, no dropout', 'mri_fs29', 'full', 'mri_fs29', 'nomri', 'mri', 0),
 ('MRI increment 3ch, dropout', 'mri3ch_fs29_drop', 'full', 'mri3ch_fs29_drop', 'nomri', 'mri', 0),
 ('MRI model (1ch, dropout) minus LR, MRI present', 'mri_fs29_drop', 'full', 'lr_full_fs29', 'full', 'mri', 0),
 ('MRI model (1ch, dropout) minus LR, MRI absent', 'mri_fs29_drop', 'full', 'lr_full_fs29', 'full', 'nomri', 0)]
@safe
def t3_pairs():
    ok = usable_mri(); rows = []
    for lab, a, ca, b, cb, sub, rep in PAIRS:
        A, B = load(a, ca, rep), load(b, cb, rep); ids = [i for i in A.index.intersection(B.index) if i in Y.index]
        if sub == 'mri': ids = [i for i in ids if i in ok]
        if sub == 'nomri': ids = [i for i in ids if i not in ok]
        y = Y.loc[ids].values; pa, pb = A.loc[ids].values, B.loc[ids].values; bs = boots(y)
        da = [AUC(y[i], pa[i]) - AUC(y[i], pb[i]) for i in bs]; dp = [AP(y[i], pa[i]) - AP(y[i], pb[i]) for i in bs]
        rows.append(dict(comparison=lab, subset=sub, repeat=('avg of 3' if rep is None else rep), n=len(y), pos=int(y.sum()), dAUROC=AUC(y, pa) - AUC(y, pb), lo=np.percentile(da, 2.5), hi=np.percentile(da, 97.5),
                         dPRAUC=AP(y, pa) - AP(y, pb), plo=np.percentile(dp, 2.5), phi=np.percentile(dp, 97.5)))
    save(pd.DataFrame(rows).round(3), 'T3_paired_differences', 'DEVELOPMENT ONLY; participant bootstrap (2000); training randomness not included; 14 comparisons, unadjusted')

@safe
def t4_sens():
    save(pd.read_csv('results/sensitivity_lr.csv').round(3), 'T4_sensitivity_variants', 'DEVELOPMENT ONLY; LR, 3x5 CV; alternative label/cohort definitions')

@safe
def t5_test():
    for f, nm in [('final_metrics', 'T5a_locked_test_metrics'), ('final_deltas', 'T5b_locked_test_paired_deltas'), ('final_full_metrics_global', 'T5c_locked_test_global'), ('final_full_metrics_operating', 'T5d_locked_test_operating_points')]:
        p = f'results/final/{f}.csv'
        if os.path.exists(p): save(pd.read_csv(p), nm, 'LOCKED TEST n=156 (36 positives), evaluated once; thresholds chosen on development data')
        else: print('missing:', p)

@safe
def t6_pooled():
    T = load('final_lr_full_fs29'); test_ids = set(T.index)
    P = pd.concat([load('lr_full_fs29'), T]); C = pd.concat([load('lr_full_clinonly'), load('final_lr_full_clinonly')])
    ids = P.index.intersection(C.index); y = Y.loc[ids].values; p, c = P.loc[ids].values, C.loc[ids].values; tag = np.array(['test' if r in test_ids else 'dev' for r in ids]); rows = []
    for sc in ['dev', 'test', 'pooled (descriptive)']:
        m = np.ones(len(y), bool) if sc.startswith('pooled') else tag == sc
        yy, pp, cc = y[m], p[m], c[m]; B = boots(yy, 1000)
        a = [AUC(yy[i], pp[i]) for i in B]; d = [AUC(yy[i], pp[i]) - AUC(yy[i], cc[i]) for i in B]
        rows.append(dict(scope=sc, n=len(yy), pos=int(yy.sum()), auroc_primary=AUC(yy, pp), lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5), auroc_clinical_only=AUC(yy, cc),
                         fs_increment=AUC(yy, pp) - AUC(yy, cc), inc_lo=np.percentile(d, 2.5), inc_hi=np.percentile(d, 97.5), prauc_primary=AP(yy, pp)))
    save(pd.DataFrame(rows).round(3), 'T6_dev_test_pooled', 'dev = repeat-averaged out-of-fold; test = locked; pooled is DESCRIPTIVE only (mixes the two)')

@safe
def t7_noncog():
    save(pd.read_csv('results/noncog/summary.csv'), 'T7a_noncognitive_models', 'DEVELOPMENT ONLY, post hoc')
    o = pd.read_csv('results/noncog/oof.csv').set_index('rid'); y = Y.loc[o.index].values; rows = []
    for a, b in [('demo+fs29', 'demo(age,sex,edu,APOE4)'), ('demo+fs29+phase', 'demo+phase'), ('fs29 only', 'demo(age,sex,edu,APOE4)')]:
        pa, pb = o[a].values, o[b].values; d = [AUC(y[i], pa[i]) - AUC(y[i], pb[i]) for i in boots(y)]
        rows.append(dict(comparison=f'{a} minus {b}', dAUROC=AUC(y, pa) - AUC(y, pb), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    save(pd.DataFrame(rows).round(3), 'T7b_noncognitive_increments', 'DEVELOPMENT ONLY, post hoc')

@safe
def t8_qc():
    l = pd.read_csv('results/mri_v2/preproc_log.csv').drop_duplicates('rid', keep='last'); l['st'] = l.status.astype(str).str[:6]
    m = elig.reset_index()[['RID', 'adni_phase', 'label_primary']].merge(l[['rid', 'st']], left_on='RID', right_on='rid', how='left'); m['st'] = m.st.fillna('NO_SCAN')
    g = m.groupby('st').label_primary.agg(n='size', positives='sum').reset_index(); g['rate'] = (g.positives / g.n).round(3)
    save(g, 'T8a_mri_qc_status', 'eligible cohort N=778; MRI QC status vs progression')
    save(pd.crosstab(m.adni_phase, m.st).reset_index(), 'T8b_mri_qc_by_phase', 'eligible cohort N=778')

def meta():
    import sklearn
    v = {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}
    try: import torch; v['torch'] = torch.__version__
    except Exception: pass
    try: v['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: v['git_commit'] = 'not a git repo'
    open(f'{OUT}/environment.txt', 'w').write('\n'.join(f'{k}: {x}' for k, x in v.items())); print('\n## environment'); print(v)
def copies():
    for f in ['results/ipcw_dev_stdout.txt', 'results/ipcw_dev_summary.csv', 'results/audit_checks_stdout.txt', 'results/baselines/summary.csv']:
        if os.path.exists(f): shutil.copy(f, OUT)
        else: print('not found (skipped):', f)
open(f'{OUT}/SCOPE_README.txt', 'w').write('''DEVELOPMENT-ONLY (n=622, participant-level CV): T2, T2b, T3, T4, T7, IPCW.
LOCKED TEST (n=156, 36 positives, evaluated once, model list frozen in experiments/final_selection.yaml): T5a-d.
POOLED dev+test (n=778): T6 'pooled' row, DESCRIPTIVE only.
POST HOC (after the locked test, dev-only): paper-scale attention, 3-channel MRI, non-cognitive models (T7), IPCW, extra metrics (T5c/d), missingness curve (T9).
''')
for fn in [t1_cohort, t2_dev, t2b_mri, t3_pairs, t4_sens, t5_test, t6_pooled, t7_noncog, t8_qc]: fn()
meta(); copies()