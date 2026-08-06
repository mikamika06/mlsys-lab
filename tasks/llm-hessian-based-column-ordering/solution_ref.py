import numpy as np

def hessian_saliency(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal Hessian salience for each input dimension of a linear layer.
    Parameters
    ----------
    W : ndarray, shape (n_out, n_in)
        Weight matrix of the linear layer.
    A : ndarray, shape (batch_size, n_in)
        Activations that were fed to the layer during inference.
    Returns
    -------
    h : ndarray, shape (n_in,)
        Salience scores for each input dimension.
    """
    n_out, n_in = W.shape
    batch_size = A.shape[0]
    h = np.empty(n_in, dtype=W.dtype)
    for j in range(n_in):
        sum_a2 = 0.0
        for i in range(batch_size):
            v = A[i, j]
            sum_a2 += v * v
        sum_w2 = 0.0
        for k in range(n_out):
            w = W[k, j]
            sum_w2 += w * w
        h[j] = sum_a2 * sum_w2
    return h
