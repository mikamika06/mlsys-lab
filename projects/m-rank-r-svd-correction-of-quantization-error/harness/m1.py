import ref
import numpy as np


def check(workdir):
    from quantcorr.svd import compute_quantization_error, svd_correction, apply_corrected_quantization

    out = {"mse_improved": 0.0}
    w = ref.TEST_WEIGHTS[0]
    w_q = np.round(w)

    err = compute_quantization_error(w, w_q)
    a, b = svd_correction(err, 4)
    w_corr = apply_corrected_quantization(w, w_q, 4)

    mse_orig = float(np.mean((w - w_q) ** 2))
    mse_corr = float(np.mean((w - w_corr) ** 2))

    if mse_corr < mse_orig:
        out["mse_improved"] = 1.0
    else:
        out["_note"] = f"SVD correction failed to improve MSE: orig={mse_orig}, corr={mse_corr}"
    return out
