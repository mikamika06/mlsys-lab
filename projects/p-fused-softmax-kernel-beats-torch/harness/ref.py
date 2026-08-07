import torch

def get_oracle_output(x):
    return torch.softmax(x, dim=-1)

def get_oracle_ratio(x):
    return 0.95
