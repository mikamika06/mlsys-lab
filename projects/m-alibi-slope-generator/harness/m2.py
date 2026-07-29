import numpy as np
import ref


def check(workdir):
    from scoremod import softcap_backward, softcap_forward

    out = {"forward_rel_err": float("inf"), "backward_rel_err": float("inf")}
    fwd_worst = 0.0
    bwd_worst = 0.0

    for x, cap in ref.SOFTCAP_CASES:
        x = np.asarray(x, dtype=np.float64)
        want_fwd = ref.softcap_forward(x, cap)
        try:
            got_fwd = np.asarray(softcap_forward(x, cap), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            out["_note"] = f"forward cap={cap} raised {type(e).__name__}: {str(e)[:120]}"
            return out
        if got_fwd.shape != want_fwd.shape:
            out["_note"] = f"forward cap={cap}: shape {got_fwd.shape} != {want_fwd.shape}"
            return out
        fwd_denom = np.maximum(np.abs(want_fwd), 1e-12)
        fwd_worst = max(fwd_worst, float(np.max(np.abs(got_fwd - want_fwd) / fwd_denom)))

        grad_output = np.ones_like(x)
        want_bwd = ref.softcap_backward(grad_output, x, cap)
        try:
            got_bwd = np.asarray(softcap_backward(grad_output, x, cap), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            out["_note"] = f"backward cap={cap} raised {type(e).__name__}: {str(e)[:120]}"
            return out
        if got_bwd.shape != want_bwd.shape:
            out["_note"] = f"backward cap={cap}: shape {got_bwd.shape} != {want_bwd.shape}"
            return out
        bwd_denom = np.maximum(np.abs(want_bwd), 1e-12)
        bwd_worst = max(bwd_worst, float(np.max(np.abs(got_bwd - want_bwd) / bwd_denom)))

    out["forward_rel_err"] = fwd_worst
    out["backward_rel_err"] = bwd_worst
    return out
