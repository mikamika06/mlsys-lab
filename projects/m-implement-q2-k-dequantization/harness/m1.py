import numpy as np
import ref


def check(workdir):
    from kquant.dequant import dequantize_q2_k, reconstruct_q3_k_scales

    out = {"max_abs_err": 1.0}
    max_err = 0.0

    for seed in range(5):
        raw_q2, _, _, _, _ = ref.generate_q2_k_block(seed=seed)
        want_q2 = ref.dequantize_q2_k_ref(raw_q2)
        got_q2 = dequantize_q2_k(raw_q2)

        err_q2 = float(np.max(np.abs(want_q2 - got_q2)))
        max_err = max(max_err, err_q2)

    rng = np.random.default_rng(123)
    hmask = rng.integers(0, 256, size=8, dtype=np.uint8).tobytes()
    scales_raw = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()

    want_q3 = ref.reconstruct_q3_k_scales_ref(hmask, scales_raw)
    got_q3 = reconstruct_q3_k_scales(hmask, scales_raw)

    err_q3 = float(np.max(np.abs(want_q3 - got_q3)))
    max_err = max(max_err, err_q3)

    out["max_abs_err"] = max_err
    return out
