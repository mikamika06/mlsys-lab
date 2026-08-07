import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity


def profile_gloo_all_reduce(tensors, num_iters=5):
    if not dist.is_initialized():
        store = dist.HashStore()
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=0,
            world_size=1,
        )

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(num_iters):
            for t in tensors:
                dist.all_reduce(t)

    return prof


def extract_all_reduce_self_time(prof):
    total_self = 0.0
    for evt in prof.key_averages():
        k = evt.key.lower()
        if "all_reduce" in k or "gloo" in k or "c10d::" in k:
            total_self += float(evt.self_cpu_time_total)
    return total_self
