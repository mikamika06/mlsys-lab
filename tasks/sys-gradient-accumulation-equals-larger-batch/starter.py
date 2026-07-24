def accumulate_grad(micro_batches, w):
    """
    Given a list of (X_i, y_i) micro-batches and a weight vector w, return
    the gradient of mean squared error loss w.r.t. w, computed by
    accumulating gradient contributions across micro-batches so the result
    equals the gradient a single large batch (the concatenation of all
    micro-batches) would produce.

    micro_batches: list of (X_i, y_i), X_i shape (b_i, D), y_i shape (b_i,).
    w: (D,) weight vector.
    Loss: L(w) = (1/N) * sum over all N examples of (X w - y)^2.
    """
    raise NotImplementedError('your code here')
