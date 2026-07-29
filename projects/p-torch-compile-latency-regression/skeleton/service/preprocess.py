import torch


def normalise(batch):
    out = []
    for row in batch:
        m = row.mean().item()
        s = row.std().item()
        out.append((row - m) / (s + 1e-6))
    return torch.stack(out)
