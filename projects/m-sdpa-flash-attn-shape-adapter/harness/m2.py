import ref
import numpy as np


def check(workdir):
    from adapter.ref_attn import reference_attention

    cases = ref.generate_cases()
    max_err = 0.0
    for q, k, v in cases:
        try:
            got_out, got_lse = reference_attention(q, k, v)
            want_out, want_lse = ref.oracle_reference_attention(q, k, v)
            err_out = float(np.max(np.abs(got_out - want_out)))
            err_lse = float(np.max(np.abs(got_lse - want_lse)))
            max_err = max(max_err, err_out, err_lse)
        except Exception as e:
            return {"max_abs_err": 999.0, "_note": str(e)}
    return {"max_abs_err": float(max_err)}
