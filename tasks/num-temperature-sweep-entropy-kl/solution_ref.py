import numpy as np

def softmax_temperature_sweep(logits: np.ndarray, temps):
    """
    Compute the temperature‑scaled softmax for each temperature in `temps`.

    Parameters
    ----------
    logits : np.ndarray
        1-D array of raw logits.
    temps : Sequence[float]
        Iterable of positive temperatures.

    Returns
    -------
    probs : np.ndarray
        2-D array of shape (len(temps), len(logits)) containing the softmax
        probabilities for each temperature. The result is dtype float64.
    """
    logits = np.asarray(logits, dtype=np.float64)
    temps = np.asarray(temps, dtype=np.float64)

    probs_list = []
    for t in temps:
        scaled = logits / t
        max_val = np.max(scaled)
        exp_shifted = np.exp(scaled - max_val)
        probs = exp_shifted / np.sum(exp_shifted)
        probs_list.append(probs)

    return np.vstack(probs_list)
