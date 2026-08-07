import numpy as np

def drop_token_0(attention_probs):
    """Drop token 0 from all queries > 0 and renormalize."""
    ablated = attention_probs.copy()
    ablated[..., 1:, 0] = 0.0
    sums = ablated.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1.0
    return ablated / sums

def measure_blowup(original_probs, ablated_probs):
    """Measure the maximum absolute difference between the probabilities."""
    return float(np.max(np.abs(original_probs - ablated_probs)))
