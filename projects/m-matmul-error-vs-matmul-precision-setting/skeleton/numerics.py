import numpy as np

def truncate_to_tf32(x: np.ndarray) -> np.ndarray:
    """
    Truncate a float32 array to TF32 precision (10 mantissa bits).
    Clear the bottom 13 bits of the float32 representation.
    Return a new contiguous float32 array.
    """
    raise NotImplementedError

def truncate_to_bf16(x: np.ndarray) -> np.ndarray:
    """
    Truncate a float32 array to bfloat16 precision (7 mantissa bits).
    Clear the bottom 16 bits of the float32 representation.
    Return a new contiguous float32 array.
    """
    raise NotImplementedError

def matmul_chain(matrices: list[np.ndarray], precision: str) -> np.ndarray:
    """
    Multiply a sequence of matrices: M0 @ M1 @ ... @ Mn
    If precision == 'tf32' or 'bf16', truncate both operands before EVERY multiplication.
    If precision == 'fp32', do not truncate.
    Return the final result array.
    """
    raise NotImplementedError

def find_unsafe_layers(layers: list[dict], x: np.ndarray, threshold: float) -> list[int]:
    """
    Evaluate each layer individually on `x`.
    For each layer, compute output in fp32.
    Then compute output in bf16 (truncate `x` and all np.ndarray parameters to bf16 before ops).

    Relative error formula: np.linalg.norm(out_bf16 - out_fp32) / (np.linalg.norm(out_fp32) + 1e-12)

    Return a list of indices of layers where this relative error is strictly greater than `threshold`.

    Supported layers:
      - {"type": "linear", "w": array, "b": array}
          out = x @ w.T + b
      - {"type": "layernorm", "gamma": array, "beta": array, "eps": float}
          mu = np.mean(x, axis=-1, keepdims=True)
          var = np.var(x, axis=-1, keepdims=True)
          out = (x - mu) / np.sqrt(var + eps) * gamma + beta
      - {"type": "relu"}
          out = np.maximum(0, x)
    """
    raise NotImplementedError
