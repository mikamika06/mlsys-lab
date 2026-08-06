import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
