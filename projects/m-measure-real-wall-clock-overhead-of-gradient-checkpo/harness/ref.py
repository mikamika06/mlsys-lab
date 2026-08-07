import torch

def get_test_inputs():
    torch.manual_seed(42)
    return torch.randn(4, 16)

def get_test_model():
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16)
    )
