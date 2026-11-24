#!/usr/bin/env python
import argparse, os, re, copy, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

P = argparse.ArgumentParser()
P.add_argument('--name', required=True)
P.add_argument('--model', default='attn', choices=['lr', 'concat', 'attn'])
P.add_argument('--clinical', default='full', choices=['full', 'nocirc'])
P.add_argument('--fs', default='fs29', choices=['none', 'fs10', 'fs29'])
P.add_argument('--mri_dir', default='')
P.add_argument('--mri_ch', type=int, default=1, choices=[1, 3])
P.add_argument('--mri_scale', type=float, default=1.0)
P.add_argument('--experts', type=int, default=1)
P.add_argument('--dim', type=int, default=64)
P.add_argument('--heads', type=int, default=4)
P.add_argument('--mask_p', type=float, default=0.0)
P.add_argument('--epochs', type=int, default=80)
P.add_argument('--lr', type=float, default=3e-4)
P.add_argument('--wd', type=float, default=1e-3)
P.add_argument('--bs', type=int, default=32)
P.add_argument('--repeats', type=int, nargs='+', default=[0, 1, 2])
P.add_argument('--phase', action='store_true')
P.add_argument('--smoke', action='store_true')
a = P.parse_args()
if a.smoke: a.epochs = 3
DEV = 'mps' if (a.mri_dir and torch.backends.mps.is_available()) else 'cpu'

# ---------------- data ----------------
coh = pd.read_csv('data/processed/cohort_v2b.csv')
coh = coh[coh.label_primary.notna()].reset_index(drop=True)
coh['y'] = coh.label_primary.astype(int)
print('eligible:', len(coh), 'positives:', coh.y.sum())
ms = pd.read_csv('data/processed/modality_status.csv').set_index('RID')
cl = pd.read_csv('data/processed/features/clinical_raw_v1.csv').set_index('RID')
C_ALL = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']
C_COLS = [c for c in C_ALL if not (a.clinical == 'nocirc' and c in ('CDRSB', 'CDGLOBAL', 'FAQTOTAL'))]
Xc = cl.reindex(coh.RID)[C_COLS].astype(float).values
if a.phase:
    ph = pd.get_dummies(coh.adni_phase).reindex(columns=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], fill_value=0)
    Xc = np.hstack([Xc, ph.astype(float).values])

FS_PATH = {'fs10': 'data/processed/features/freesurfer_raw_v1.csv',
           'fs29': 'data/processed/features/freesurfer29_raw_v1.csv'}
if a.fs != 'none':
    fsd = pd.read_csv(FS_PATH[a.fs]).set_index('RID')
    fcols = [c for c in fsd.columns if re.fullmatch(r'ST\d+(SV|TA|CV)', c)]
    if not fcols: raise SystemExit(f'No ST* columns found. Columns: {list(fsd.columns)}')
    R = fsd.reindex(coh.RID)
    Xf = R[fcols].astype(float).copy()
    if 'ST10CV' in R:
        for c in fcols:
            if c.endswith('SV'): Xf[c] = Xf[c] / R['ST10CV'] * 1000
    Xf = Xf.values
    stat = ms.reindex(coh.RID)['freesurfer_status']
    print('FS columns:', len(fcols), '| FS status:', stat.value_counts().to_dict())
    ok = stat.isin(['OBSERVED', 'PARTIAL']).values & ~np.isnan(Xf).all(1)
else:
    Xf, ok = np.zeros((len(coh), 1)), np.zeros(len(coh), bool)
m_fs = ok.astype(np.float32)

MR = {}
if a.mri_dir:
    for r in coh.RID:
        p = f'{a.mri_dir}/{r}.npy'
        if os.path.exists(p): MR[r] = np.load(p)
    if not MR: raise SystemExit('mri_dir given but no <RID>.npy files found')
m_mri = np.array([r in MR for r in coh.RID], np.float32)
print('masks: clinical 1.0 | FS', m_fs.mean().round(3), '| MRI', m_mri.mean().round(3))
Y, RIDS = coh.y.values, coh.RID.values
idx = {r: i for i, r in enumerate(RIDS)}
M0 = np.stack([np.ones(len(coh), np.float32), m_fs, m_mri], 1)

sp = pd.read_csv('data/processed/splits/splits_v1.csv')
dev = sp[sp.split != 'test']
def folds(rep):
    d = dev[dev.repeat == rep]
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        if g.split.nunique() > 1: g = g[g.split.str.lower().str.startswith('val')]
        val = g.rid.values
        if not 100 < len(val) < 150:
            raise SystemExit(f'Unexpected val size {len(val)}; split values: {dev.split.unique()}\n{dev.head()}')
        yield f, np.setdiff1d(dev.rid.unique(), val), val

def fit_prep(trn):
    med = np.nanmedian(Xc[trn], 0); Zc = np.where(np.isnan(Xc), med, Xc)
    Zc = (Zc - Zc[trn].mean(0)) / (Zc[trn].std(0) + 1e-6)
    if a.fs != 'none':
        t = trn[m_fs[trn] > 0]; medf = np.nanmedian(Xf[t], 0); Zf = np.where(np.isnan(Xf), medf, Xf)
        Zf = (Zf - Zf[t].mean(0)) / (Zf[t].std(0) + 1e-6); Zf[m_fs == 0] = 0
    else: Zf = Xf.copy()
    return Zc.astype(np.float32), Zf.astype(np.float32)

# ---------------- model ----------------
class MLPEnc(nn.Module):
    def __init__(s, d_in, d, p=(.3, .2, .1)):
        super().__init__(); dims = [d_in, 128, 256, d]; L = []
        for i in range(3): L += [nn.Linear(dims[i], dims[i+1]), nn.LayerNorm(dims[i+1]), nn.GELU(), nn.Dropout(p[i])]
        s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x)

class CNN3D(nn.Module):
    def __init__(s, c_in, d, w=(16, 32, 64, 96)):
        super().__init__(); L = []; c = c_in
        for o in w:
            L += [nn.Conv3d(c, o, 3, stride=2, padding=1), nn.GroupNorm(4, o), nn.GELU(),
                  nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU()]; c = o
        s.f = nn.Sequential(*L)
        s.out = nn.Sequential(nn.Linear(2*c, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(.1))
    def forward(s, x):
        h = s.f(x); return s.out(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1))

def make_channels(x, ch):
    if a.mri_scale != 1.0: x = F.interpolate(x, scale_factor=a.mri_scale, mode='trilinear', align_corners=False)
    if ch == 1: return x
    dx = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 0, 0, 1))
    dz = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1))
    g = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
    m = F.avg_pool3d(x, 5, 1, 2); v = (F.avg_pool3d(x*x, 5, 1, 2) - m*m).clamp_min(0)
    return torch.cat([x, g, v], 1)

class Fusion(nn.Module):
    def __init__(s, dc, df, mode, d, h, K, ch, use_mri, use_fs):
        super().__init__(); s.mode = mode
        s.enc_c = MLPEnc(dc, d)
        s.enc_f = MLPEnc(df, d) if use_fs else None
        s.enc_m = CNN3D(ch, d) if use_mri else None
        s.temb = nn.Parameter(torch.randn(3, d) * .02)
        s.attn = nn.MultiheadAttention(d, h, dropout=.1, batch_first=True)
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ffn = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(.1), nn.Linear(2*d, d))
        H = 2*d
        s.fuse = nn.Sequential(nn.Linear(3*d + 3, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(.4),
                               nn.Linear(H, H//2), nn.LayerNorm(H//2), nn.GELU(), nn.Dropout(.3))
        s.experts = nn.ModuleList([nn.Sequential(nn.Linear(H//2, H//4), nn.GELU(), nn.Linear(H//4, 1)) for _ in range(K)])
        s.gate = nn.Linear(H//2, K) if K > 1 else None
    def forward(s, xc, xf, mri, m):
        ec = s.enc_c(xc)
        ef = s.enc_f(xf) if s.enc_f is not None else torch.zeros_like(ec)
        em = torch.zeros_like(ec)
        if s.enc_m is not None and mri is not None:
            em[m[:, 2] > .5] = s.enc_m(mri)
        T = (torch.stack([ec, ef, em], 1) + s.temb) * m.unsqueeze(-1)
        w = None
        if s.mode == 'attn':
            att, w = s.attn(T, T, T, key_padding_mask=(m < .5), need_weights=True, average_attn_weights=True)
            T = s.ln1(T + att); T = s.ln2(T + s.ffn(T)) * m.unsqueeze(-1)
        z = s.fuse(torch.cat([T.flatten(1), m], 1))
        o = torch.cat([e(z) for e in s.experts], 1)
        logit = (torch.softmax(s.gate(z), 1) * o).sum(1) if s.gate is not None else o[:, 0]
        return logit, w

def mri_batch(bi, m, train):
    sel = np.where(m[:, 2] > 0)[0]
    if len(sel) == 0: return None
    x = torch.from_numpy(np.stack([MR[RIDS[bi[j]]] for j in sel]).astype(np.float32)).to(DEV)
    if train: x = x * torch.empty(len(sel), 1, 1, 1, 1, device=DEV).uniform_(.9, 1.1)
    return make_channels(x, a.mri_ch)

def fwd(model, bi, Zc, Zf, m, train):
    m = m.copy()
    if model.enc_f is None: m[:, 1] = 0
    if model.enc_m is None: m[:, 2] = 0
    mri = mri_batch(bi, m, train) if model.enc_m is not None else None
    mt = torch.from_numpy(m).to(DEV)
    lg, w = model(torch.from_numpy(Zc[bi]).to(DEV), torch.from_numpy(Zf[bi]).to(DEV), mri, mt)
    return lg, w, mt

def drop(m):
    if a.mask_p > 0:
        m = m.copy(); m[:, 1:] *= ~(np.random.rand(len(m), 2) < a.mask_p)
    return m

@torch.no_grad()
def predict(model, rows, Zc, Zf, cond):
    model.eval(); out, aw = [], []
    for i in range(0, len(rows), 32):
        bi = rows[i:i+32]; m = M0[bi].copy()
        if cond in ('nofs', 'clin'): m[:, 1] = 0
        if cond in ('nomri', 'clin'): m[:, 2] = 0
        lg, w, mt = fwd(model, bi, Zc, Zf, m, False)
        out.append(torch.sigmoid(lg).cpu().numpy())
        if w is not None:
            mq = mt.unsqueeze(2); aw.append(((w * mq).sum(1) / mq.sum(1)).cpu().numpy())
    return np.concatenate(out), (np.concatenate(aw) if aw else None)

def run_lr(Zc, Zf, trn, val):
    def X(rows, on):
        if a.fs == 'none': return Zc[rows]
        f = Zf[rows] if on else np.zeros_like(Zf[rows]); k = m_fs[rows, None] if on else np.zeros((len(rows), 1), np.float32)
        return np.hstack([Zc[rows], f, k])
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring='roc_auc', max_iter=3000).fit(X(trn, True), Y[trn])
    res = {'full': (clf.predict_proba(X(val, True))[:, 1], None)}
    if a.fs != 'none': res['nofs'] = (clf.predict_proba(X(val, False))[:, 1], None)
    return res

def run_fold(trn, val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Zc, Zf = fit_prep(trn)
    if a.model == 'lr': return run_lr(Zc, Zf, trn, val)
    itr, iva = train_test_split(trn, test_size=.15, stratify=Y[trn], random_state=seed)
    model = Fusion(Zc.shape[1], Zf.shape[1], a.model, a.dim, a.heads, a.experts, a.mri_ch,
                   bool(MR), a.fs != 'none').to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    pw = torch.tensor((len(itr) - Y[itr].sum()) / Y[itr].sum(), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, bstate, bad = 1e9, None, 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(itr)
        for i in range(0, len(perm), a.bs):
            bi = perm[i:i+a.bs]
            if len(bi) < 4: continue
            lg, _, _ = fwd(model, bi, Zc, Zf, drop(M0[bi]), True)
            loss = lossf(lg, torch.from_numpy(Y[bi]).float().to(DEV))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vl = 0
            for i in range(0, len(iva), 32):
                bi = iva[i:i+32]; lg, _, _ = fwd(model, bi, Zc, Zf, M0[bi], False)
                vl += lossf(lg, torch.from_numpy(Y[bi]).float().to(DEV)).item() * len(bi)
            vl /= len(iva)
        if vl < best - 1e-4: best, bstate, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if ep >= 10 and bad >= 12: break
    model.load_state_dict(bstate)
    conds = ['full'] + (['nofs'] if a.fs != 'none' else []) + (['nomri'] if MR else []) + (['clin'] if (a.fs != 'none' or MR) else [])
    return {c: predict(model, val, Zc, Zf, c) for c in conds}

# ---------------- run ----------------
os.makedirs(f'results/deep/{a.name}', exist_ok=True)
recs, t0 = [], time.time()
for rep in ([0] if a.smoke else a.repeats):
    for f, trn_r, val_r in folds(rep):
        if a.smoke and f > 1: break
        trn = np.array([idx[r] for r in trn_r]); val = np.array([idx[r] for r in val_r])
        res = run_fold(trn, val, 1000 * rep + int(f))
        for cond, (p, aw) in res.items():
            for j, i in enumerate(val):
                row = dict(rid=RIDS[i], repeat=rep, fold=int(f), cond=cond, prob=float(p[j]))
                if aw is not None: row.update(a_clin=aw[j, 0], a_fs=aw[j, 1], a_mri=aw[j, 2])
                recs.append(row)
        print(f'rep {rep} fold {f} done | {time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(recs); df.to_csv(f'results/deep/{a.name}/oof.csv', index=False)
json.dump({**vars(a), 'torch': torch.__version__}, open(f'results/deep/{a.name}/config.json', 'w'), indent=1)
info = coh[['RID', 'y', 'adni_phase']].assign(mri=m_mri, fs=m_fs)
def ci(y, p, fn, n=500):
    rng = np.random.default_rng(0); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])
out = []
for cond, g in df.groupby('cond'):
    rep_auc = [AUC(info.set_index('RID').loc[gg.rid, 'y'], gg.prob) for _, gg in g.groupby('repeat')]
    pr = g.groupby('rid').prob.mean().rename('p').reset_index().merge(info, left_on='rid', right_on='RID')
    subs = {'all': pr.RID > 0, 'ADNI1': pr.adni_phase == 'ADNI1', 'ADNIGO+2': pr.adni_phase.isin(['ADNIGO', 'ADNI2']), 'ADNI3': pr.adni_phase == 'ADNI3'}
    if MR: subs.update({'mri_present': pr.mri == 1, 'mri_absent': pr.mri == 0})
    for s, mk in subs.items():
        d = pr[mk]
        if d.y.nunique() < 2: continue
        lo, hi = ci(d.y.values, d.p.values, AUC)
        out.append(dict(cond=cond, subset=s, n=len(d), pos=int(d.y.sum()), auroc=AUC(d.y, d.p), auroc_lo=lo, auroc_hi=hi,
                        prauc=AP(d.y, d.p), brier=brier_score_loss(d.y, d.p),
                        logloss=log_loss(d.y, np.clip(d.p, 1e-6, 1-1e-6)),
                        rep_auroc_mean=np.mean(rep_auc) if s == 'all' else np.nan, rep_auroc_sd=np.std(rep_auc) if s == 'all' else np.nan))
S = pd.DataFrame(out); S.to_csv(f'results/deep/{a.name}/summary.csv', index=False)
print(S.round(3).to_string(index=False))