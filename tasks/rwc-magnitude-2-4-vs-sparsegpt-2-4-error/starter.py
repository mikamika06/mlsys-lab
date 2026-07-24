import numpy as np


def compare_magnitude_vs_sparsegpt_2_4(W: np.ndarray, X: np.ndarray):
    """Compare naive magnitude 2:4 pruning against Hessian-aware SparseGPT
    2:4 pruning, by the relative Frobenius-norm output error each
    introduces on the linear layer Y = X @ W.T.

    W: (out_dim, in_dim) float weight matrix; in_dim is a multiple of 4.
    X: (s, in_dim) float calibration activations.

    1. Y_true = X @ W.T.

    2. Magnitude 2:4: for each row of W, for each consecutive group of 4
       columns, zero the 2 smallest-|value| entries (keep the 2 largest,
       unmodified, no compensation). err_magnitude = relative Frobenius
       error of X @ W_hat_mag.T vs Y_true:
           ||Y_approx - Y_true||_F / ||Y_true||_F

    3. SparseGPT 2:4 (Hessian-aware): damped Hessian
           H = X.T @ X / s + 1e-4 * I
       Hinv = inverse of H. For each row of W, for each group of 4
       columns, saliency score for column c is w[r,c]**2 / Hinv[c,c];
       prune the 2 lowest-saliency columns in the group, keep the other
       2. For each pruned column c (weight value `old` before zeroing),
       compensate every still-kept column k in the SAME group:
           w[r,k] -= old * Hinv[k,c] / Hinv[c,c]
       then zero w[r,c]. err_sparsegpt = same relative Frobenius error
       using this compensated, pruned weight matrix.

    4. reduction = 1 - err_sparsegpt / err_magnitude.

    Returns (err_magnitude, err_sparsegpt, reduction).
    """
    raise NotImplementedError('your code here')
