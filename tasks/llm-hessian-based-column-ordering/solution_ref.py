def hessian_saliency(W: list[list[float]], A: list[list[float]]) -> list[float]:
    """
    Compute the diagonal Hessian salience for each input dimension of a linear layer.

    Parameters
    ----------
    W : list of list of float
        Weight matrix of the linear layer, shape (n_out, n_in).
    A : list of list of float
        Activations that were fed to the layer during inference, shape (batch_size, n_in).

    Returns
    -------
    h : list of float
        Salience scores for each input dimension, length n_in.
    """
    n_out = len(W)
    n_in = len(W[0])
    batch_size = len(A)
    h = [0.0] * n_in
    for j in range(n_in):
        sum_a2 = 0.0
        for i in range(batch_size):
            v = A[i][j]
            sum_a2 += v * v
        sum_w2 = 0.0
        for k in range(n_out):
            w = W[k][j]
            sum_w2 += w * w
        h[j] = sum_a2 * sum_w2
    return h
