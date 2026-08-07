import torch


def run_qlora_training(model, steps=20):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        x = torch.randn(4, 16)
        target = torch.zeros(4, 16)
        out = model(x)
        loss = ((out - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    return steps
