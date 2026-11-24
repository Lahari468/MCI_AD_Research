#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
M="--mri_dir data/interim/mri_mni2mm_v2 --mri_ch 1 --mri_scale 0.67 --bs 8 --lr 1e-4"

"$PY" src/training/train_cv.py --name mri_fs29_drop --model attn --fs fs29 $M --mask_p 0.3 --repeats 0
"$PY" src/training/train_cv.py --name mri_fs29 --model attn --fs fs29 $M --repeats 0
"$PY" src/training/train_cv.py --name mri_nofs_drop --model attn --fs none $M --mask_p 0.3 --repeats 0
