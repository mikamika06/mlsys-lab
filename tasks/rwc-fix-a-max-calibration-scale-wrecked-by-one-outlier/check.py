import numpy as np


def _oracle(x, qmax, percentile):
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.percentile(np.abs(x), percentile))
    scale = amax / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    reconstructed = q * scale
    return amax, scale, reconstructed


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    0.12,
                    -0.24,
                    0.35,
                    0.18,
                    -0.42,
                    0.31,
                    100.0,
                ],
                dtype=np.float64,
            ),
            127,
            99.0,
        ),
        (
            np.array(
                [
                    -1.2,
                    0.8,
                    1.5,
                    -0.4,
                    0.2,
                    75.0,
                    0.6,
                    -0.9,
                ],
                dtype=np.float64,
            ),
            127,
            95.0,
        ),
        (
            np.array([0.1, -0.2, 0.4, 0.7, 1.1, 30.0], dtype=np.float64),
            63,
            90.0,
        ),
    ]

    max_abs_err = 0.0
    reconstruction_rel_err = 0.0

    for x, qmax, percentile in cases:
        ref_amax, ref_scale, ref_recon = _oracle(x, qmax, percentile)
        try:
            got_amax, got_scale, got_recon = sol.calibrate_scale_and_dequantize(
                x.copy(), qmax=qmax, percentile=percentile
            )
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "reconstruction_rel_err": float("inf"),
            }

        max_abs_err = max(max_abs_err, abs(float(got_amax) - ref_amax))
        got_recon = np.asarray(got_recon, dtype=np.float64)
        reconstruction_rel_err = max(
            reconstruction_rel_err,
            float(
                np.linalg.norm(got_recon - ref_recon)
                / (np.linalg.norm(x) + 1e-12)
            ),
        )

    return {
        "max_abs_err": max_abs_err,
        "reconstruction_rel_err": reconstruction_rel_err,
    }
