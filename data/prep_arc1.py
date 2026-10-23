"""Build data/arc1.npz for ARC-AGI-1 training in the URM/HRM recipe (github.com/UbiquantAI/URM data/build_arc_dataset.py).

Subsets: ARC-AGI-1 training + evaluation + ConceptARC (corpus); every demonstration pair of every task is a training
pair, and the evaluation tasks' test pairs are the test set (transductive, as URM). Augmentation is done on the fly by
lt.data.ArcTrainDataset (version v = 0..num_aug of a task: dihedral transform and a permutation of colors 1..9, fixed by
(seed, task, v); training pairs also get a random translation inside the 30x30 canvas). Raw canvas values: 0 pad,
1..10 colors 0..9, 11 = EOS (the row and column just outside the grid); lt.data adds 1 (tokens 1 pad = ignored with
ignore_label0, 2..11 colors, 12 EOS), so num_symbols = 11.

Stored: arc_train_in / arc_train_out [N, 30, 30] uint8 (color + 1, 0 outside the grid) with arc_train_hw [N, 4]
(h_in, w_in, h_out, w_out) and arc_train_task [N]; test_inputs / test_labels as finished canvases (top-left, no
augmentation) with arc_test_task; arc_num_tasks.

    python data/prep_arc1.py ARC_AGI_REPO CONCEPTARC_REPO
    python data/prep_arc1.py --kaggle KAGGLE/combined/arc-agi --trm-split --out data/arc1_trm.npz

--kaggle PREFIX reads TRM's input format instead (PREFIX_{training,evaluation,concept}_{challenges,solutions}.json, as
TinyRecursiveModels dataset/build_arc_dataset.py --input-file-prefix kaggle/combined/arc-agi); tasks keep the order of
data/arc1.tasks.txt. --trm-split is TRM's split (--test-set-name evaluation): the test pairs of the training and concept
tasks are training pairs too (3,092 + 896 = 3,988 per epoch); the test set is unchanged (419 evaluation test pairs).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

N = 30


def canvas(grid, eos=True):
    """Raw canvas of a grid at the top-left: color + 1, EOS (11) on the row and column just outside, 0 elsewhere."""
    g = np.asarray(grid, dtype=np.uint8)
    out = np.zeros((N, N), dtype=np.uint8)
    h, w = g.shape
    out[:h, :w] = g + 1
    if eos:
        if h < N:
            out[h, :min(w + 1, N)] = 11
        if w < N:
            out[:min(h + 1, N), w] = 11
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arc', nargs='?')
    ap.add_argument('concept', nargs='?')
    ap.add_argument('--kaggle', help='TRM input prefix (.../combined/arc-agi) instead of the two repositories')
    ap.add_argument('--trm-split', action='store_true', help='training/concept test pairs are training pairs (TRM)')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'arc1.npz'))
    opt = ap.parse_args()
    if opt.kaggle:
        order = (Path(__file__).resolve().parent / 'arc1.tasks.txt').read_text().split()
        tasks = {}
        for sub in ('training', 'evaluation', 'concept'):
            ch = json.load(open(f'{opt.kaggle}_{sub}_challenges.json'))
            so = json.load(open(f'{opt.kaggle}_{sub}_solutions.json'))
            for k, v in ch.items():
                tasks[f'{sub}/{k}'] = dict(train=v['train'], test=[dict(input=t['input'], output=o)
                                                                  for t, o in zip(v['test'], so[k])])
        if sorted(tasks) != sorted(order):
            raise SystemExit('kaggle tasks differ from data/arc1.tasks.txt')
        subsets = [(sub, [(n.split('/', 1)[1], tasks[n]) for n in order if n.startswith(sub + '/')])
                   for sub in ('training', 'evaluation', 'concept')]
    else:
        subsets = [(sub, [(Path(f).stem, json.load(open(f))) for f in files]) for sub, files in
                   (('training', sorted(glob.glob(f'{opt.arc}/data/training/*.json'))),
                    ('evaluation', sorted(glob.glob(f'{opt.arc}/data/evaluation/*.json'))),
                    ('concept', sorted(glob.glob(f'{opt.concept}/corpus/*/*.json'))))]
    tr_in, tr_out, tr_hw, tr_t, te_in, te_out, te_t, names = [], [], [], [], [], [], [], []
    for subset, items in subsets:
        for stem, task in items:
            t = len(names)
            names.append(f'{subset}/{stem}')
            pairs = task['train'] + (task['test'] if opt.trm_split and subset != 'evaluation' else [])
            for p in pairs:
                i, o = np.asarray(p['input']), np.asarray(p['output'])
                a, b = np.zeros((N, N), np.uint8), np.zeros((N, N), np.uint8)
                a[:i.shape[0], :i.shape[1]] = i + 1
                b[:o.shape[0], :o.shape[1]] = o + 1
                tr_in.append(a); tr_out.append(b); tr_hw.append((*i.shape, *o.shape)); tr_t.append(t)
            if subset == 'evaluation':
                for p in task['test']:
                    te_in.append(canvas(p['input'])); te_out.append(canvas(p['output'])); te_t.append(t)
    np.savez_compressed(opt.out, arc_train_in=np.array(tr_in), arc_train_out=np.array(tr_out),
                        arc_train_hw=np.array(tr_hw, dtype=np.uint8), arc_train_task=np.array(tr_t, dtype=np.int32),
                        test_inputs=np.array(te_in), test_labels=np.array(te_out), arc_test_task=np.array(te_t, dtype=np.int32),
                        arc_num_tasks=np.int64(len(names)), num_symbols=np.int64(11))
    meta = dict(tasks=len(names), per_subset={s: len(fs) for s, fs in subsets}, train_pairs=len(tr_in), test_pairs=len(te_in),
                split='trm (training/concept test pairs trained)' if opt.trm_split else 'demonstrations only',
                source=('TRM kaggle/combined/arc-agi_{training,evaluation,concept}' if opt.kaggle else
                        'fchollet/ARC-AGI data/{training,evaluation} + victorvikram/ConceptARC corpus'))
    Path(opt.out).with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    Path(opt.out).with_suffix('.tasks.txt').write_text('\n'.join(names) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
