"""Error accumulation in generator words (theory doc 4.4) on learned generators: evaluate a trained RuleSynapseModel
with code- and cell-space composition after an orthogonal perturbation of every generator, W_g -> W_g expm(eps S_g)
(S_g random skew-symmetric with unit operator norm, so ||W_g' - W_g||_op ~ eps). k = 3 exact accuracy by group.

    python -m toy_g.rs_perturb runs/rs/A_learned_code_d32_s1
"""
import json
import sys
from pathlib import Path

import torch

from . import data as D
from .rule_synapse import RuleSynapseModel
from .train_rs import eval_involutive


def main(run_dir, eps_list=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), seeds=3):
    cfg = json.loads((Path(run_dir) / 'config.json').read_text())
    sd = torch.load(Path(run_dir) / 'final.pt', map_location='cpu')
    inst = D.load_instances()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    for compose in ('code', 'cell'):
        m = RuleSynapseModel(mode=cfg['mode'], d=cfg['d'], cmap='perm', compose=compose).to(device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models[compose] = m
    W0 = models['code'].gen_matrices().detach()
    d = W0.shape[-1]
    print(f'{run_dir}: d = {d}; rows: eps, then k=3 exact (g0 / g1 / g2) for code and cell composition '
          f'(mean over {seeds} perturbation draws)')
    for eps in eps_list:
        acc = {c: torch.zeros(3) for c in models}
        for s in range(seeds if eps > 0 else 1):
            g = torch.Generator().manual_seed(s)
            S = torch.randn(4, d, d, generator=g)
            S = S - S.transpose(1, 2)
            S = S / torch.linalg.matrix_norm(S, ord=2)[:, None, None]
            Wp = W0 @ torch.matrix_exp(eps * S.to(device))
            for c, m in models.items():
                m.gen_matrices = lambda Wp=Wp: Wp
                r = eval_involutive(m, inst, device)
                acc[c] += torch.tensor([r['g0'], r['g1'], r['g2']])
        n = seeds if eps > 0 else 1
        print(f'eps {eps:4.2f}  code ' + ' / '.join(f'{100 * v:5.1f}' for v in acc['code'] / n) +
              '   cell ' + ' / '.join(f'{100 * v:5.1f}' for v in acc['cell'] / n), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
