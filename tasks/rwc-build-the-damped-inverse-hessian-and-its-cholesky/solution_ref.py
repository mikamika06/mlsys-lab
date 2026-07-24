import numpy as np


def damped_inv_hessian_cholesky(X: np.ndarray, damp_pct: float) -> dict:
    """Build GPTQ's layer-wise Hessian proxy from calibration activations,
    damp it, invert it, and return the UPPER-triangular Cholesky factor
    of the inverse -- the exact quantity GPTQ's sequential column update
    consumes.

    X        : (n_cal, d_in) calibration activations for one linear
               layer's input.
    damp_pct : float in (0, 1), e.g. 0.01. Damping strength as a fraction
               of the mean diagonal.

    H = 2 X^T X                              (d_in, d_in)
    H_damped = H + damp_pct * mean(diag(H)) * I
    Hinv = H_damped^{-1}
    U = upper-triangular factor with Hinv = U^T U
      (U = cholesky(Hinv).T -- transposing the standard LOWER Cholesky
       factor L of Hinv, since Hinv = L L^T means Hinv = (L^T)^T (L^T)
       too, and L^T is upper triangular).

    Returns {"H": H_damped, "Hinv": Hinv, "U": U}, each (d_in, d_in).
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]

    H = 2.0 * (X.T @ X)
    diag_mean = float(np.mean(np.diag(H)))
    damp = damp_pct * diag_mean
    H_damped = H + damp * np.eye(d)

    Hinv = np.linalg.inv(H_damped)
    L = np.linalg.cholesky(Hinv)
    U = L.T

    return {"H": H_damped, "Hinv": Hinv, "U": U}
