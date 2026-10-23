"""NCCL sanity check for scripts/arc1_trm.sh setup: torchrun --standalone --nproc_per_node=N scripts/nccl_check.py"""
import os

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
x = torch.ones(16 * 2 ** 20, device='cuda')                   # 64 MB, more than one step's gradient + embedding traffic
dist.all_reduce(x)
torch.cuda.synchronize()
assert int(x[0].item()) == dist.get_world_size()
if dist.get_rank() == 0:
    print(f'NCCL all_reduce ok on {dist.get_world_size()} GPUs', flush=True)
dist.destroy_process_group()
