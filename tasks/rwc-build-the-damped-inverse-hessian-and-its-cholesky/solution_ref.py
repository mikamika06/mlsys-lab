import math
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
    n_cal, d = X.shape

    H = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(n_cal):
                s += X[k, i] * X[k, j]
            H[i][j] = 2.0 * s

    diag_sum = 0.0
    for i in range(d):
        diag_sum += H[i][i]
    diag_mean = diag_sum / d

    damp = damp_pct * diag_mean

    H_damped = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            val = H[i][j]
            if i == j:
                val += damp
            H_damped[i][j] = val

    augmented = [[H_damped[i][j] for j in range(d)] + [1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

    for i in range(d):
        pivot_row = i
        max_val = abs(augmented[i][i])
        for r in range(i + 1, d):
            val = abs(augmented[r][i])
            if val > max_val:
                max_val = val
                pivot_row = r

        if pivot_row != i:
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

        factor = augmented[i][i]
        for c in range(2 * d):
            augmented[i][c] /= factor

        for r in range(d):
            if r != i:
                f = augmented[r][i]
                if f != 0.0:
                    for c in range(2 * d):
                        augmented[r][c] -= f * augmented[i][c]

    Hinv = [[augmented[i][j + d] for j in range(d)] for i in range(d)]

    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k]
            if i == j:
                val = Hinv[i][i] - s
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (Hinv[i][j] - s) / L[j][j]

    U = [[L[j][i] for j in range(d)] for i in range(d)]

    return {
        "H": np.array(H_damped, dtype=np.float64),
        "Hinv": np.array(Hinv, dtype=np.float64),
        "U": np.array(U, dtype=np.float64),
    }
