import torch

def inspect_autocast_dtypes(model, x, dtype=torch.bfloat16):
    raise NotImplementedError

def verify_weights_unchanged(model, x, dtype=torch.bfloat16):
    raise NotImplementedError

def check_overflow(layer, x):
    raise NotImplementedError
