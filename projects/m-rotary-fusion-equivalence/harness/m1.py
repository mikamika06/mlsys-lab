import ref
import numpy as np

def check(workdir):
    from rotary.fusion import apply_fused_rotary
    q, k, cos, sin = ref.generate_test_inputs()
    got_q, got_k = apply_fused_rotary(q, k, cos, sin)
    want_q, want_k = ref.ref_apply_fused_rotary(q, k, cos, sin)

    err_q = np.max(np.abs(got_q - want_q))
    err_k = np.max(np.abs(got_k - want_k))
    max_err = float(max(err_q, err_k))

    out = {"max_abs_err": max_err}
    return out
