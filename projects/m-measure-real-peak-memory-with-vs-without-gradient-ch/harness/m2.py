import ref
import torch

def check(workdir):
    from gradckpt.flow import check_gradient_flow_frozen
    torch.manual_seed(42)
    model = ref.ToyModel(hidden_dim=32, num_layers=4)
    x = torch.randn(2, 32)
    got = check_gradient_flow_frozen(model, x)
    out = {"flow_reproduced": 0.0}
    if got is True or (isinstance(got, dict) and got.get("failed") is True):
        out["flow_reproduced"] = 1.0
    else:
        out["_note"] = f"expected gradient flow failure, got {got}"
    return out
