import ref
import torch

def check(workdir):
    from graphbreak.cond import safe_conditional
    x = torch.tensor(4.0)
    y = torch.tensor(5.0)
    pred = torch.tensor(True)
    try:
        got = safe_conditional(pred, x, y)
        want = ref.evaluate_cond(pred, x, y)
        if torch.allclose(got, want):
            return {"cond_matched": 1.0}
    except Exception as e:
        return {"cond_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    return {"cond_matched": 0.0}
