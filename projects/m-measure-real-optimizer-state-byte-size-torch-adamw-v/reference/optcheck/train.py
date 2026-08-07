import torch
import torch.nn as nn


def run_short_training(model, optimizer_cls, steps=10, **kwargs):
    optimizer = optimizer_cls(model.parameters(), **kwargs)
    criterion = nn.MSELoss()
    losses = []
    torch.manual_seed(42)
    for _ in range(steps):
        optimizer.zero_grad()
        x = torch.randn(4, model.in_features)
        target = torch.ones(4, 1)
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
