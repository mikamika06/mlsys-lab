import torch
import ref


def check(workdir):
    from quantutil.core import compute_error
    torch.manual_seed(42)
    t = torch.randn(32, 32) * 2.0
    t[5, :] *= 30.0

    got_t = compute_error(t, "per-tensor")
    got_r = compute_error(t, "per-row")

    want_t = ref.compute_error(t, "per-tensor")
    want_r = ref.compute_error(t, "per-row")

    ok = (abs(got_t - want_t) < 1e-5) and (abs(got_r - want_r) < 1e-5) and (got_r < got_t)
    return {"errors_matched": 1.0 if ok else 0.0}
