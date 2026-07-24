import numpy as np


def compare_q4_errors(block: np.ndarray) -> dict:
    x = np.asarray(block, dtype=np.float64)

    amax = np.max(np.abs(x))
    s_sym = 2 * amax / 15
    q_sym = np.clip(np.round(x / s_sym), -8, 7)
    recon_sym = s_sym * q_sym
    err_sym = np.linalg.norm(recon_sym - x) / (np.linalg.norm(x) + 1e-12)

    xmin = np.min(x)
    xmax = np.max(x)
    s_aff = (xmax - xmin) / 15
    z = np.round(-xmin / s_aff)
    q_aff = np.clip(np.round(x / s_aff + z), 0, 15)
    recon_aff = s_aff * (q_aff - z)
    err_aff = np.linalg.norm(recon_aff - x) / (np.linalg.norm(x) + 1e-12)

    return {
        "q4_0_error": float(err_sym),
        "affine_int4_error": float(err_aff),
        "winner": "q4_0" if err_sym <= err_aff else "affine_int4",
    }
