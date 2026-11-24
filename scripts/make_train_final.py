s = open('src/training/train_cv.py').read()

old1 = "P.add_argument('--smoke', action='store_true')\n"
new1 = old1 + "P.add_argument('--final', action='store_true')\nP.add_argument('--seeds', type=int, nargs='+', default=[0])\n"

old2 = ("for rep in ([0] if a.smoke else a.repeats):\n"
        "    for f, trn_r, val_r in folds(rep):\n"
        "        if a.smoke and f > 1: break\n")

new2 = ("def _iter():\n"
        "    if a.final:\n"
        "        assert os.environ.get('FINAL_EVAL') == '1', 'Locked test: set FINAL_EVAL=1'\n"
        "        te = sp[sp.split == 'test'].rid.values\n"
        "        for sd in a.seeds: yield sd, 0, dev.rid.unique(), te\n"
        "    else:\n"
        "        for rep in ([0] if a.smoke else a.repeats):\n"
        "            for f, trn_r, val_r in folds(rep): yield rep, f, trn_r, val_r\n"
        "for rep, f, trn_r, val_r in _iter():\n"
        "    if True:\n"
        "        if a.smoke and f > 1: break\n")

assert s.count(old1) == 1 and s.count(old2) == 1, 'train_cv.py differs from expected; inspect loop section'
open('src/training/train_final.py', 'w').write(s.replace(old1, new1).replace(old2, new2))
print('wrote src/training/train_final.py')
