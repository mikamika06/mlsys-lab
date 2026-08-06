import torch


def make_test_data(seed: int = 42):
    torch.manual_seed(seed)
    x = torch.randn(8, 8, dtype=torch.float32)
    alpha = 0.75
    return x, alpha
