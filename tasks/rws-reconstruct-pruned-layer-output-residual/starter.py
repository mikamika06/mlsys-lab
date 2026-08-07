def apply_wanda_mask(W: list[list[float]], M: list[list[float]], X: list[list[float]]):
    """
    Apply the (precomputed) binary pruning mask M to W and report both
    the pruned layer's output and the output residual it introduces:
      Y = (W ⊙ M) @ X
      R = W @ X - Y
    W, M: (d_out, d_in). X: (d_in, n). Returns (Y, R), each (d_out, n).
    """
    raise NotImplementedError('your code here')
