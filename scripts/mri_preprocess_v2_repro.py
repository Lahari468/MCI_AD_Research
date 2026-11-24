#!/usr/bin/env python
import os, glob, shutil, subprocess, tempfile, argparse, time
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
import numpy as np, pandas as pd, nibabel as nib, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
FSL = os.environ.get('FSLDIR')
HEAD = f'{FSL}/data/standard/MNI152_T1_2mm.nii.gz'
MASK = f'{FSL}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
OUT, FAILDIR, QC, LOG = 'data/interim/mri_mni2mm_v2_repro', 'data/interim/mri_mni2mm_v2_repro_qcfail', 'results/mri_v2_repro/qc', 'results/mri_v2_repro/preproc_log.csv'
ALT = ['-cost', 'normcorr', '-searchrx', '-180', '180', '-searchry', '-180', '180', '-searchrz', '-180', '180']

def sh(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    if r.returncode: raise RuntimeError(f"{cmd[0]}: {r.stderr.decode(errors='ignore')[-200:]}")

def index_dicom(roots):
    ix = {}
    for r in roots:
        for dp, dn, _ in os.walk(r, followlinks=True):
            for d in dn:
                if d[0] == 'I' and d[1:].isdigit(): ix[d[1:]] = os.path.join(dp, d)
    return ix

def register(t, src, tag, extra):
    sh(['flirt', '-in', src, '-ref', HEAD, '-out', f'{t}/{tag}.nii.gz', '-omat', f'{t}/{tag}.mat',
        '-dof', '12', '-interp', 'trilinear'] + extra)
    x = nib.load(f'{t}/{tag}.nii.gz').get_fdata().astype(np.float32)
    ref = nib.load(HEAD).get_fdata().astype(np.float32); mk = nib.load(MASK).get_fdata() > 0
    ncc = float(np.corrcoef(x[mk], ref[mk])[0, 1])
    A = np.loadtxt(f'{t}/{tag}.mat')[:3, :3]
    scale = float(abs(np.linalg.det(A)) ** (1 / 3)); U, _, Vt = np.linalg.svd(A); R = U @ Vt
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return x, ncc, scale, ang

def is_bad(ncc, scale, ang): return (not 0.7 < scale < 1.4) or ang > 45 or ncc < 0.2

def process(rid, iid, dpath):
    if os.path.exists(f'{OUT}/{rid}.npy') or os.path.exists(f'{FAILDIR}/{rid}.npy'): return rid, iid, 'skip', {}
    t = tempfile.mkdtemp(prefix=f'mri{rid}_'); t0 = time.time()
    try:
        sh(['dcm2niix', '-z', 'y', '-i', 'y', '-f', 'img', '-o', t, dpath])
        nii = sorted(glob.glob(f'{t}/*.nii.gz'), key=os.path.getsize)[-1]
        im0 = nib.load(nii)
        if im0.ndim > 3: nib.save(nib.Nifti1Image(np.asarray(im0.dataobj)[..., 0], im0.affine), f'{t}/v.nii.gz'); nii = f'{t}/v.nii.gz'
        sh(['fslreorient2std', nii, f'{t}/ro.nii.gz'])
        sh(['robustfov', '-i', f'{t}/ro.nii.gz', '-r', f'{t}/crop.nii.gz'])
        img = sitk.ReadImage(f'{t}/crop.nii.gz', sitk.sitkFloat32)
        msk = sitk.OtsuThreshold(img, 0, 1, 200)
        n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([30, 20, 10])
        n4.Execute(sitk.Shrink(img, [4] * 3), sitk.Shrink(msk, [4] * 3))
        sitk.WriteImage(img / sitk.Exp(n4.GetLogBiasFieldAsImage(img)), f'{t}/n4.nii.gz')
        x, ncc, sc, ang = register(t, f'{t}/n4.nii.gz', 'a', []); retry = 0
        if is_bad(ncc, sc, ang):
            x2, n2, s2, a2 = register(t, f'{t}/n4.nii.gz', 'b', ALT); retry = 1
            if n2 > ncc: x, ncc, sc, ang = x2, n2, s2, a2
        mk = nib.load(MASK).get_fdata() > 0
        lo, hi = np.percentile(x[mk], [0.5, 99.5]); xc = np.clip(x, lo, hi)
        z = np.where(mk, (xc - xc[mk].mean()) / (xc[mk].std() + 1e-6), 0).astype(np.float32)
        flag = 'FAIL' if is_bad(ncc, sc, ang) else ('REVIEW' if ncc < 0.3 else 'OK')
        d = OUT if flag != 'FAIL' else FAILDIR; os.makedirs(d, exist_ok=True)
        np.save(f'{d}/{rid}.npy', z[None].astype(np.float16))
        fig, a_ = plt.subplots(1, 3, figsize=(7, 2.5))
        for k, s in enumerate([z[45], z[:, 54], z[:, :, 45]]): a_[k].imshow(s.T, cmap='gray', origin='lower'); a_[k].axis('off')
        fig.suptitle(f'{rid} {flag} ncc={ncc:.2f} scale={sc:.2f} rot={ang:.0f}', fontsize=8); fig.savefig(f'{QC}/{rid}.png', dpi=70); plt.close(fig)
        return rid, iid, flag, dict(ncc=ncc, scale=sc, angle=ang, retry=retry, secs=int(time.time() - t0))
    except Exception as e:
        return rid, iid, 'ERROR: ' + str(e)[-200:].replace('\n', ' '), {}
    finally: shutil.rmtree(t, ignore_errors=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--rids', nargs='+', type=int, default=[]); ap.add_argument('--workers', type=int, default=3); a = ap.parse_args()
    assert FSL and all(shutil.which(c) for c in ('flirt', 'robustfov', 'dcm2niix', 'fslreorient2std')), 'Need FSL and dcm2niix on PATH'
    os.makedirs(QC, exist_ok=True)
    sel = pd.read_csv('data/processed/mri_selected_eligible.csv'); sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
    sel['dpath'] = sel.iid.map(index_dicom(a.roots))
    if a.rids: sel = sel[sel.RID.isin(a.rids)]
    todo = sel[sel.dpath.notna()]; print('selected', len(sel), '| found on disk', len(todo), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, int(r.RID), r.iid, r.dpath) for r in todo.itertuples()]
        for n, f in enumerate(as_completed(futs), 1):
            rid, iid, st, m = f.result()
            if st != 'skip':
                pd.DataFrame([dict(rid=rid, image_id=iid, status=st, **m)]).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
            print(n, len(todo), rid, st, m, flush=True)