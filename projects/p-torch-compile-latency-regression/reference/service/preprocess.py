import torch


def normalise(batch):
    m = batch.mean(dim=-1, keepdim=True)
    s = batch.std(dim=-1, keepdim=True)
    return (batch - m) / (s + 1e-6)
