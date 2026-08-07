import math

def kv_quant_error_propagation(Q: list[list[float]], K: list[list[float]], V: list[list[float]], K_hat: list[list[float]], V_hat: list[list[float]], scale: float | None=None) -> dict[str, float]:
    """Compute how KV quantization error propagates through softmax attention.

    Returns dict with keys: output_mse, kv_error, amplification.
    """
    raise NotImplementedError('your code here')
