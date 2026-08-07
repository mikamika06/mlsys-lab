import numpy as np
import ref


def check(workdir):
    try:
        from rectatt.attention import sdpa_rectangular_causal, flash_attn_sim
    except Exception as e:
        return {"max_abs_err": 1.0, "_note": f"Import error: {e}"}

    np.random.seed(1337)
    max_err = 0.0

    for n_q, n_kv in ref.TEST_SHAPES:
        q = np.random.randn(2, 4, n_q, 32)
        k = np.random.randn(2, 4, n_kv, 32)
        v = np.random.randn(2, 4, n_kv, 32)

        for align in ["bottom_right", "top_left"]:
            try:
                ref_sdpa = ref.sdpa_rectangular_causal(q, k, v, alignment=align)
                got_sdpa = sdpa_rectangular_causal(q, k, v, alignment=align)
                err_sdpa = np.max(np.abs(ref_sdpa - got_sdpa))
                max_err = max(max_err, float(err_sdpa))

                ref_fa = ref.flash_attn_sim(q, k, v, is_causal=True, alignment=align)
                got_fa = flash_attn_sim(q, k, v, is_causal=True, alignment=align)
                err_fa = np.max(np.abs(ref_fa - got_fa))
                max_err = max(max_err, float(err_fa))

                fa_sdpa_err = np.max(np.abs(got_fa - got_sdpa))
                max_err = max(max_err, float(fa_sdpa_err))
            except Exception as e:
                return {"max_abs_err": 1.0, "_note": f"Execution error for shape ({n_q}, {n_kv}): {e}"}

    return {"max_abs_err": max_err}
