import numpy as np
import ref


def check(workdir):
    try:
        from rectatt.probe import compute_causal_mask, compute_offset
    except Exception as e:
        return {"max_abs_err": 1.0, "_note": f"Import error: {e}"}

    max_err = 0.0
    for n_q, n_kv in ref.TEST_SHAPES:
        for align in ["bottom_right", "top_left"]:
            try:
                ref_off = ref.compute_offset(n_q, n_kv, align)
                got_off = compute_offset(n_q, n_kv, align)
                if ref_off != got_off:
                    return {"max_abs_err": 1.0, "_note": f"Offset mismatch for {n_q},{n_kv},{align}: got {got_off}, want {ref_off}"}

                ref_m = ref.compute_causal_mask(n_q, n_kv, align)
                got_m = compute_causal_mask(n_q, n_kv, align)

                err = np.max(np.abs(ref_m.astype(float) - got_m.astype(float)))
                max_err = max(max_err, float(err))
            except Exception as e:
                return {"max_abs_err": 1.0, "_note": f"Error during execution: {e}"}

    return {"max_abs_err": max_err}
