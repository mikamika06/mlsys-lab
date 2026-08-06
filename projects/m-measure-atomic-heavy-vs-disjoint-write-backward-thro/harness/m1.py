import ref
import torch


def check(workdir):
    out = {"correctness_matched": 0.0}
    try:
        from triton_bw import kernels
        torch.manual_seed(42)
        x = torch.randn(128, 128)
        grad = torch.randn(128, 128)
        res_atomic = kernels.atomic_heavy_backward(x, grad)
        res_disjoint = kernels.disjoint_write_backward(x, grad)
        if res_atomic is not None and res_disjoint is not None:
            if torch.allclose(res_atomic, res_disjoint, atol=1e-5, rtol=1e-5):
                out["correctness_matched"] = 1.0
            else:
                out["_note"] = "atomic and disjoint outputs do not match"
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
