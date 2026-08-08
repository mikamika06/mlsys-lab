import torch


def inspect_autocast_state():
    return {"enabled": torch.is_autocast_enabled(), "dtype": str(torch.get_autocast_gpu_dtype())}
