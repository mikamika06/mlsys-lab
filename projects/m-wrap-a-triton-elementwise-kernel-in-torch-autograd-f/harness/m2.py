import torch


def check(workdir):
    from triton_wrap.autograd import FusedSiluAutograd

    out = {"gradcheck_passed": 0.0}
    x = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
    try:
        res = torch.autograd.gradcheck(FusedSiluAutograd.apply, x, eps=1e-6, atol=1e-4)
        if res:
            out["gradcheck_passed"] = 1.0
    except Exception as e:
        out["_note"] = f"gradcheck raised exception: {type(e).__name__}: {str(e)[:100]}"
    return out
