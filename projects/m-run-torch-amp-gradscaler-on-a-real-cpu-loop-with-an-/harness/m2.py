import ref
import torch

def check(workdir):
    from scaler_lab.skipped_step import verify_skipped_step
    torch.manual_seed(123)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 4)
    try:
        res = verify_skipped_step(model, opt, x)
        if res is True:
            return {"params_unchanged": 1.0}
    except Exception as e:
        return {"params_unchanged": 0.0, "_note": f"failed: {e}"}
    return {"params_unchanged": 0.0}
