import numpy as np

def block_rescale_softmax(logits: np.ndarray, block_size: int) -> np.ndarray:
    """
    Compute softmax of `logits` using a block‑wise rescaling strategy.
    The result is identical to the monolithic softmax computed with NumPy.
    """
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.size
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    # Global maximum for numerical stability
    M = np.max(logits)
    exp_scaled = np.empty_like(logits)
    denom = 0.0
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = logits[start:end]
        m_b = np.max(block)
        scale_factor = np.exp(m_b - M)          # exp(local_max - global_max)
        exp_block = np.exp(block - m_b) * scale_factor
        exp_scaled[start:end] = exp_block
        denom += exp_block.sum()
    return exp_scaled / denom
