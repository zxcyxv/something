"""Per-row CE along the trajectory of the trained Rule 110 model: is the loss monotone after the horizon?"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1]); from r110lib import *
torch.set_grad_enabled(False)
model, cfg = build(f'{sys.argv[1]}/model.pt')
x, y = sample(4096, torch.Generator(device=dev).manual_seed(7)); L = []
run(model, x, 128, lambda b, h, lg, box, st: L.append(F.cross_entropy(lg.reshape(-1, 3), (y + 1).reshape(-1), reduction='none').view(x.shape[0], -1).sum(1)))
L = torch.stack(L)                                    # [128, B] row CE (sum over cells)
SH = [2, 4, 6, 8, 10, 12, 16, 24, 32, 64, 128]
print('block                 ' + ' '.join(f'{b:>7}' for b in SH))
print('median row CE         ' + ' '.join(f'{float(L[b-1].median()):7.4f}' for b in SH))
print('rows with CE rising % ' + ' '.join(f'{float((L[b-1] > L[b-2]).float().mean())*100:7.1f}' for b in SH))
print('rows CE < log2 %      ' + ' '.join(f'{float((L[b-1] < 0.6931).float().mean())*100:7.1f}' for b in SH))
amin = L.argmin(0) + 1
print('block of minimum CE: quantiles', [int(v) for v in amin.float().quantile(torch.tensor([.1, .25, .5, .75, .9], device=dev))])
