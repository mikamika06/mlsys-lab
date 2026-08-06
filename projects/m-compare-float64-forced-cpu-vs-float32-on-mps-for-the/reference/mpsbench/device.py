import torch


def check_mps_support():
    is_built = getattr(torch.backends.mps, "is_built", lambda: False)()
    is_avail = getattr(torch.backends.mps, "is_available", lambda: False)() if is_built else False
    return {"is_built": bool(is_built), "is_available": bool(is_avail)}
