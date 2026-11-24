import glob, numpy as np, pandas as pd
def rd(p):
    f = sorted(glob.glob(p))
    if not f: raise SystemExit(f'no file matches {p} (check: ls data/raw/clinical data/raw/freesurfer)')
    return pd.read_csv(f[0], low_memory=False)
coh = pd.read_csv('data/processed/cohort_v2b.csv'); coh = coh[coh.label_primary.notna()].copy(); coh['T0'] = pd.to_datetime(coh.T_anchor)
sp = pd.read_csv('data/processed/splits/splits_v1.csv').drop_duplicates('rid').set_index('rid'); coh['split'] = coh.RID.map(sp.split)
phc = rd('data/raw/clinical/ADSP_PHC_DEMODX.csv'); phc['d'] = pd.to_datetime(phc.EXAMDATE, errors='coerce'); phc = phc[phc.d.notna() & phc.PHC_Diagnosis.notna()]
PH = {r: (g.d.values, g.PHC_Diagnosis.values) for r, g in phc.groupby('RID')}
def prep(p, col, date='VISDATE'):
    d = rd(p); d['d'] = pd.to_datetime(d[date], errors='coerce'); d = d[d.d.notna() & (d[col] >= 0)]
    return {r: g.d.values for r, g in d.groupby('RID')}
CL = {'MMSE': prep('data/raw/clinical/MMSE_*.csv', 'MMSCORE'), 'CDR': prep('data/raw/clinical/CDR_*.csv', 'CDRSB'),
      'FAQ': prep('data/raw/clinical/FAQ_*.csv', 'FAQTOTAL'), 'ADAS': prep('data/raw/clinical/ADAS_*.csv', 'TOTSCORE')}
FS = prep('data/raw/freesurfer/UCSFFSX7_*.csv', 'ST29SV', 'EXAMDATE')
def has(a, lo, hi): return a is not None and bool(((a >= lo.to_datetime64()) & (a <= hi.to_datetime64())).any())
EMPTY = (np.array([], dtype='datetime64[ns]'), np.array([]))
for L in (180, 365):
    rows = []
    for r in coh.itertuples():
        lm = r.T0 + pd.Timedelta(days=L); dts, dx = PH.get(r.RID, EMPTY)
        ad = bool(((dts <= lm.to_datetime64()) & (dx == 3)).any())
        mci = bool(((dts >= (lm - pd.Timedelta(days=90)).to_datetime64()) & (dts <= lm.to_datetime64()) & (dx == 2)).any())
        rows.append(dict(split=r.split, y=int(r.label_primary), ad_by_L=ad, at_risk=(not ad) and mci,
                         clin=all(has(CL[k].get(r.RID), lm - pd.Timedelta(days=90), lm) for k in CL),
                         fs=has(FS.get(r.RID), lm - pd.Timedelta(days=180), lm)))
    d = pd.DataFrame(rows); a = d[d.at_risk]; c = a[a.clin]; cf = c[c.fs]
    print(f'\n=== landmark = baseline + {L} days (endpoint unchanged: AD by day 730) ===')
    print(f'positives already AD by landmark (no longer at risk): {int(((d.y == 1) & d.ad_by_L).sum())} of {int(d.y.sum())}')
    for nm, x in [('at risk (MCI at landmark, AD-free)', a), ('+ complete clinical set at landmark', c), ('+ FreeSurfer at landmark', cf)]:
        dv, te = x[x.split != 'test'], x[x.split == 'test']
        print(f'{nm}: n={len(x)} pos={int(x.y.sum())} | dev n={len(dv)} pos={int(dv.y.sum())} | test n={len(te)} pos={int(te.y.sum())}')