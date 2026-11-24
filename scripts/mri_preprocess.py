#!/usr/bin/env python
import os, sys, glob, shutil, subprocess, tempfile, argparse, time
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
import numpy as np, pandas as pd, nibabel as nib, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
FSL = os.environ.get('FSLDIR')
MNI = f'{FSL}/data/standard/MNI152_T1_2mm_brain.nii.gz' if FSL else None
OUT, FAILDIR, QC = 'data/interim/mri_mni2mm', 'data/interim/mri_mni2mm_qcfail', 'results/mri/qc'

def sh(cmd): subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def index_dicom(roots):
    ix = {}
    for r in roots:
        for dp, dn, _ in os.walk(r, followlinks=True):
            for d in dn:
                if d[0] == 'I' and d[1:].isdigit(): ix[d[1:]] = os.path.join(dp, d)
    return ix

def process(rid, iid, dpath):
    if os.path.exists(f'{OUT}/{rid}.npy') or os.path.exists(f'{FAILDIR}/{rid}.npy'): return rid, iid, 'skip', {}
    t = tempfile.mkdtemp(prefix=f'mri{rid}_'); t0 = time.time()
    try:
        sh(['dcm2niix', '-z', 'y', '-f', 'img', '-o', t, dpath])
        nii = sorted(glob.glob(f'{t}/*.nii.gz'), key=os.path.getsize)[-1]
        sh(['fslreorient2std', nii, f'{t}/ro.nii.gz'])
        img = sitk.ReadImage(f'{t}/ro.nii.gz', sitk.sitkFloat32)
        msk = sitk.OtsuThreshold(img, 0, 1, 200)
        n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([30, 20, 10])
        n4.Execute(sitk.Shrink(img, [4]*3), sitk.Shrink(msk, [4]*3))
        sitk.WriteImage(img / sitk.Exp(n4.GetLogBiasFieldAsImage(img)), f'{t}/n4.nii.gz')
        sh(['bet', f'{t}/n4.nii.gz', f'{t}/brain', '-R', '-f', '0.35'])
        sh(['flirt', '-in', f'{t}/brain.nii.gz', '-ref', MNI, '-out', f'{t}/mni.nii.gz',
            '-omat', f'{t}/a.mat', '-dof', '12', '-interp', 'trilinear'])
        x = nib.load(f'{t}/mni.nii.gz').get_fdata().astype(np.float32)
        ref = nib.load(MNI).get_fdata().astype(np.float32); rm = ref > 0; mk = x > 0
        z = np.where(mk, (x - x[mk].mean()) / (x[mk].std() + 1e-6), 0).astype(np.float32)
        ncc = float(np.corrcoef(x[rm], ref[rm])[0, 1]); dice = float(2 * (mk & rm).sum() / (mk.sum() + rm.sum()))
        flag = 'FAIL' if (dice < .75 or ncc < .4) else ('REVIEW' if dice < .88 else 'OK')
        d = OUT if flag != 'FAIL' else FAILDIR; os.makedirs(d, exist_ok=True)
        np.save(f'{d}/{rid}.npy', z[None].astype(np.float16))
        fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
        for k, s in enumerate([z[45], z[:, 54], z[:, :, 45]]): ax[k].imshow(s.T, cmap='gray', origin='lower'); ax[k].axis('off')
        fig.suptitle(f'{rid} {flag} dice={dice:.2f} ncc={ncc:.2f}', fontsize=8); fig.savefig(f'{QC}/{rid}.png', dpi=70); plt.close(fig)
        return rid, iid, flag, dict(ncc=ncc, dice=dice, vol_mm3=int(mk.sum() * 8), secs=int(time.time() - t0))
    except Exception as e:
        return rid, iid, 'ERROR: ' + (getattr(e, 'stderr', b'') or str(e).encode())[-200:].decode(errors='ignore'), {}
    finally: shutil.rmtree(t, ignore_errors=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    assert FSL and all(shutil.which(c) for c in ('bet', 'flirt', 'dcm2niix', 'fslreorient2std')), 'Need FSL (FSLDIR, bet, flirt) and dcm2niix on PATH'
    os.makedirs(QC, exist_ok=True); os.makedirs('results/mri', exist_ok=True)
    sel = pd.read_csv('data/processed/mri_selected_eligible.csv')
    coh = pd.read_csv('data/processed/cohort_v2b.csv')
    coh = coh[coh.label_primary.notna()]
    sel = sel[sel.RID.isin(coh.RID)].copy()
    print('Eligible MRI records:', len(sel))
    sel['iid'] = sel.image_id.astype(str).str.replace(r'\D', '', regex=True)
    sel['dpath'] = sel.iid.map(index_dicom(a.roots))
    print('selected', len(sel), '| found on disk', sel.dpath.notna().sum(), flush=True)
    todo = sel[sel.dpath.notna()]; todo = todo.head(a.limit) if a.limit else todo
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, int(r.RID), r.iid, r.dpath) for r in todo.itertuples()]
        for n, f in enumerate(as_completed(futs), 1):
            rid, iid, st, m = f.result()
            pd.DataFrame([dict(rid=rid, image_id=iid, status=st, **m)]).to_csv('results/mri/preproc_log.csv', mode='a', header=not os.path.exists('results/mri/preproc_log.csv'), index=False)
            print(n, len(todo), rid, st, m, flush=True)