import ref
import torch
import numpy as np

def check(workdir):
    out = {"bwd_max_abs_err": 999.0}
    try:
        from custom_op.rbf import rbf_interact
    except ImportError:
        out["_note"] = "could not import rbf_interact"
        return out

    x_np, y_np, gamma = ref.get_fixtures(B=2, N=5, M=4, D=3, seed=123)
    x_t = torch.from_numpy(x_np).requires_grad_(True)
    y_t = torch.from_numpy(y_np).requires_grad_(True)

    try:
        fwd_out = rbf_interact(x_t, y_t, gamma)
        ref_out, ref_gx, ref_gy, g_np = ref.oracle_rbf(x_np, y_np, gamma)

        g_t = torch.from_numpy(g_np)
        fwd_out.backward(g_t)

        err_x = np.max(np.abs(x_t.grad.numpy() - ref_gx))
        err_y = np.max(np.abs(y_t.grad.numpy() - ref_gy))

        out["bwd_max_abs_err"] = float(max(err_x, err_y))
    except Exception as e:
        out["_note"] = f"backward failed: {type(e).__name__}: {e}"

    return out
