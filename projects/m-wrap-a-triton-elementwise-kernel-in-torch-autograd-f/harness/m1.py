import ref
import torch


def check(workdir):
    from triton_wrap.kernel import fused_silu_backward

    out = {"grad_match": 0.0}
    x = torch.randn(32, 32, dtype=torch.float32)
    go = torch.ones_like(x)
    want = ref.compute_analytic_grad(x, go)
    got = fused_silu_backward(x, go)
    if torch.allclose(got, want, atol=1e-5, rtol=1e-5):
        out["grad_match"] = 1.0
    else:
        out["_note"] = "analytic gradient does not match reference"
    return out
