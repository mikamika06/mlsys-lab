import torch

def sample_target_fn(x):
    if x.sum() > 0:
        print("branch taken")
    return x * 2.0

def get_sample_args():
    return (torch.randn(4),)
