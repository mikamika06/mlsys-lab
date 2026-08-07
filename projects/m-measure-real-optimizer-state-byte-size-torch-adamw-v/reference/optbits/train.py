import torch
import torch.nn as nn


def train_short_loop(model, x, y, steps=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    initial_loss = None
    final_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        val = loss.item()
        if step == 0:
            initial_loss = val
        final_loss = val
    return initial_loss, final_loss
