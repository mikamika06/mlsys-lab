import torch
import torch.nn as nn
import ref

def check(workdir):
    from packutil.train import run_dummy_finetune
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(16, 16))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [torch.randn(8, 16) for _ in range(10)]
    res = run_dummy_finetune(model, optimizer, data)
    match = 1.0 if res.get("loss_decreased") else 0.0
    out = {"loss_decreased": match}
    if match == 0.0:
        out["_note"] = "loss did not decrease"
    return out
