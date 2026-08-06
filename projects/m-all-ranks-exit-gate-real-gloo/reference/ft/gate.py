import torch.distributed as dist
import datetime

def sync_or_exit(rank: int, world_size: int, store_path: str, timeout_sec: float) -> bool:
    try:
        store = dist.FileStore(store_path, world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_sec)
        )
        dist.barrier()
        return True
    except Exception:
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
