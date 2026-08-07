import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(x)


def get_reference_measure(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch_b = total * 8
    adam8_b = total * 2 + max(1, total // 256) * 4
    return {
        "torch_adamw": float(torch_b),
        "adamw_8bit": float(adam8_b),
        "size_ratio": float(adam8_b / torch_b),
    }


def get_reference_train(model, x, y):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    o0 = crit(model(x), y).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        optimizer.step()
    o1 = crit(model(x), y).item()
    return o0, o1


def get_reference_memory(model):
    res = {}
    for name, p in model.named_parameters():
        n = p.numel()
        res[name] = {
            "numel": n,
            "adam32bit_bytes": n * 8,
            "adam8bit_bytes": n * 2 + max(1, n // 256) * 4,
        }
    return res
