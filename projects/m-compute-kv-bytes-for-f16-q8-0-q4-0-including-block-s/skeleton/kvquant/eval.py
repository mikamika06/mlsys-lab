import numpy as np


def quantize_dequantize(x: np.ndarray, dtype: str) -> np.ndarray:
    """Quantize and immediately dequantize array x along blocks of size 32."""
    raise NotImplementedError


def eval_needle_retrieval(
    seq_len: int = 512,
    head_dim: int = 64,
    needle_idx: int = 128,
    dtype: str = "q4_0",
    seed: int = 42,
) -> dict:
    """Run needle-in-haystack retrieval evaluation on quantized KV cache."""
    raise NotImplementedError
