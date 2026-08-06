import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def calculate_min_world_size(param_bytes, optimizer_bytes_per_param, grad_bytes_per_param, activation_bytes, budget_bytes):
    for ws in range(1, 1024):
        per_rank_params = param_bytes / ws
        per_rank_grads = grad_bytes_per_param * (param_bytes / ws)
        per_rank_opt = optimizer_bytes_per_param * (param_bytes / ws)
        total = per_rank_params + per_rank_grads + per_rank_opt + activation_bytes
        if total <= budget_bytes:
            return ws
    return 1024

def _worker(rank, world_size, info_file):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
    model = FSDP(model)
    sizes = [p.numel() for p in model.parameters()]
    if rank == 0:
        with open(info_file, "w") as f:
            f.write(str({rank: sizes}))
    else:
        with open(info_file, "a") as f:
            f.write(str({rank: sizes}))
    dist.destroy_process_group()

def launch_and_get_sharded_sizes():
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    info_file = tf.name
    try:
        mp.spawn(_worker, args=(2, info_file), nprocs=2, join=True)
        with open(info_file, "r") as f:
            content = f.read()
        return eval(content)
    finally:
        if os.path.exists(info_file):
            os.unlink(info_file)

def verify_model_weights(original_model, sharded_model):
    orig_state = original_model.state_dict()
    shrd_state = sharded_model.state_dict()
    for name, param in orig_state.items():
        if name not in shrd_state:
            return False
        if not torch.allclose(param, shrd_state[name], atol=1e-5, rtol=1e-5):
            return False
    return True

CONFIGS = [
    (1000000, 16, 4, 100000, 500000),
    (50000000, 12, 4, 2000000, 20000000)
]
