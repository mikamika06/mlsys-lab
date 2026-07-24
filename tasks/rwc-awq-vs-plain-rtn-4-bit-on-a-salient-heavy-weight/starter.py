import numpy as np


def compare_awq_rtn_error(W: np.ndarray, X: np.ndarray):
    """Compare plain RTN INT4 quantization vs AWQ-scaled INT4 quantization
    of a linear layer's weight, on a weight/activation pair where some
    input channels are much larger in magnitude ("salient") than others.

    W: (out_dim, in_dim) float weight matrix.
    X: (batch, in_dim) float activations.
    Y_true = X @ W.T is the exact float output.

    Symmetric INT4 row-wise (per output channel) round-to-nearest
    quantization of a matrix V:
        delta = max(|V[i, :]|) / 7   (per row i; use 1e-9 if the row is all zero)
        V_hat[i, :] = clip(round(V[i, :] / delta), -8, 7) * delta

    1. err_rtn: quantize W directly with the formula above, compute
       relative Frobenius error of X @ W_hat.T vs Y_true.
    2. AWQ: compute the per-input-channel scale s_j = mean_b |X[b, j]|
       (average activation magnitude on channel j), scale W by s before
       quantizing (W * s, broadcasting s over columns), quantize with the
       same INT4 formula, then divide back by s. err_awq is the relative
       Frobenius error of this reconstruction's output vs Y_true.
    3. reduction = 1 - err_awq / err_rtn.

    Relative Frobenius error of an approximation Y_approx vs Y_true:
        ||Y_approx - Y_true||_F / ||Y_true||_F

    Returns (err_rtn, err_awq, reduction).
    """
    raise NotImplementedError('your code here')
