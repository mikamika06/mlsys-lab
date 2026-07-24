import numpy as np

def compare_pi_ntk(base_logits: np.ndarray,
                   pi_logits: np.ndarray,
                   ntk_logits: np.ndarray) -> tuple[float, float]:
    """
    Compute the mean KL divergence between each of two sets of logits and a base set.
    The implementation is fully vectorised and uses only NumPy operations.
    """
    # Ensure float64 for numerical stability
    base = np.asarray(base_logits, dtype=np.float64)
    pi   = np.asarray(pi_logits, dtype=np.float64)
    ntk  = np.asarray(ntk_logits, dtype=np.float64)

    def _mean_kl(a: np.ndarray, b: np.ndarray) -> float:
        # Softmax with numerical stability
        a_max = a.max(axis=-1, keepdims=True)
        b_max = b.max(axis=-1, keepdims=True)
        p = np.exp(a - a_max)
        q = np.exp(b - b_max)
        p /= p.sum(axis=-1, keepdims=True)
        q /= q.sum(axis=-1, keepdims=True)

        # KL divergence per row
        kl_rows = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(axis=-1)
        return float(kl_rows.mean())

    pi_kl = _mean_kl(base, pi)
    ntk_kl = _mean_kl(base, ntk)

    return pi_kl, ntk_kl
