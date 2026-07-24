import numpy as np


def sample_sequence(logits: np.ndarray, temperature: float, seed: int) -> np.ndarray:
    """Reproduce the seeded temperature-sampled id sequence.

    Args:
        logits: float64 array of shape (T, V), one logit row per decode step.
        temperature: positive float tau.
        seed: integer seed for np.random.default_rng.

    Returns:
        int64 array of shape (T,) with the sampled token id at each step.
    """
    raise NotImplementedError("your code here")
