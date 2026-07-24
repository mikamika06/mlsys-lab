import numpy as np


def compare_prune_quant_order(W: np.ndarray, X: np.ndarray, group_size: int, sparsity: float, bits: int = 4):
    """Compare reconstruction MSE of two prune+quantize pipelines, per group
    of `group_size` columns, using a Wanda-style importance score
    S_ij = |W_ij| * ||X[:, j]||_2 to choose which weights to keep.

    W: shape (rows, cols), cols an exact multiple of group_size.
    X: shape (n_samples, cols), calibration activations.
    sparsity: fraction of each group to prune (keep the top 1 - sparsity
      fraction, ranked by score, at least 1 element).
    bits: symmetric quantization bit width (qmax = 2**(bits-1) - 1).

    Order A (prune-then-quant): zero out the pruned weights first, then
      derive the quantization scale from the max |value| of the SURVIVING
      (kept) weights only, and quantize/dequantize.
    Order B (quant-then-prune): derive the quantization scale from the max
      |value| of the FULL (unpruned) group, quantize/dequantize everything,
      then zero out the pruned positions.

    Both are compared to the original, unpruned W to get a mean squared
    error over the whole tensor.

    Returns (mse_prune_then_quant, mse_quant_then_prune).
    """
    raise NotImplementedError('your code here')
