import ref
import torch
import numpy as np

def check(workdir):
    out = {"fwd_max_abs_err": 999.0, "compile_success": 0.0}
    try:
        from custom_op.rbf import rbf_interact
    except ImportError:
        out["_note"] = "could not import rbf_interact"
        return out

    x_np, y_np, gamma = ref.get_fixtures()
    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)

    try:
        fwd_out = rbf_interact(x_t, y_t, gamma)
        ref_out, _, _, _ = ref.oracle_rbf(x_np, y_np, gamma)
        out["fwd_max_abs_err"] = float(np.max(np.abs(fwd_out.cpu().numpy() - ref_out)))
    except Exception as e:
        out["_note"] = f"forward failed: {type(e).__name__}: {e}"
        return out

    try:
        @torch.compile(backend="aot_eager", fullgraph=True)
        def compiled_fwd(a, b, g):
            return rbf_interact(a, b, g)

        _ = compiled_fwd(x_t, y_t, gamma)
        out["compile_success"] = 1.0
    except Exception as e:
        out["_note"] = f"compile failed (fake kernel missing or wrong shape?): {type(e).__name__}: {e}"

    return out
