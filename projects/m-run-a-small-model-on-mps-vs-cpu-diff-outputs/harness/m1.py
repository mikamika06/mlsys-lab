import ref
import torch

def check(workdir):
    from mpscheck.compare import compare_outputs
    model = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)
    want = ref.compare_outputs(model, x)
    try:
        got = compare_outputs(model, x)
    except Exception as e:
        return {"max_abs_err": 999.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}
    err = abs(want - got)
    return {"max_abs_err": float(err)}
