for CL in full nocirc; do
  python src/training/train_cv.py --name lr_${CL}_fs29          --model lr     --clinical $CL --fs fs29
  python src/training/train_cv.py --name concat_${CL}_fs29      --model concat --clinical $CL --fs fs29
  python src/training/train_cv.py --name concatdrop_${CL}_fs29  --model concat --clinical $CL --fs fs29 --mask_p 0.3
  python src/training/train_cv.py --name attn_${CL}_fs29        --model attn   --clinical $CL --fs fs29
  python src/training/train_cv.py --name attndrop_${CL}_fs29    --model attn   --clinical $CL --fs fs29 --mask_p 0.3
  python src/training/train_cv.py --name attndrop_k5_${CL}_fs29 --model attn   --clinical $CL --fs fs29 --mask_p 0.3 --experts 5
done