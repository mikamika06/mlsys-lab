import ref
import torch


def check(workdir):
    from leakdiag.eval import check_activation_retention
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    inputs = torch.randn(4, 16)
    res = check_activation_retention(model, inputs)
    out = {"retention_detected": 0.0}
    if res["cleared_with_nograd"] == 1.0 and res["retained_without_nograd"] == 1.0:
        out["retention_detected"] = 1.0
    else:
        out["_note"] = f"Activation check failed: {res}"
    return out
