import numpy as np


def gptq_quantize_layer(W: np.ndarray, X: np.ndarray, nbits: int, damp: float):
    """
    Full GPTQ column-by-column quantization of W with H^-1 error
    compensation, in natural (left-to-right) column order, as described in
    task.md. H = X^T X is the calibration Hessian.

    Returns (Wq, mse):
      Wq  -- (d_out, d_in) quantized weight matrix.
      mse -- float, mean((X @ Wq^T - X @ W^T)**2), the layer output error.
    """
    raise NotImplementedError('your code here')
