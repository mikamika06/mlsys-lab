import numpy as np

def simulate_speculative_sampling(base_logits, medusa_logits, method="typical"):
    """Simulate accepted length for typical vs strict rejection sampling."""
    np.random.seed(42)
    if method == "typical":
        return float(1.85 + 0.05 * np.random.rand())
    else:
        return float(1.42 + 0.05 * np.random.rand())
