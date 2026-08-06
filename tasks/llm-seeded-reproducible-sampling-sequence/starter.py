import math
import random

def sample_sequence(logits: list[list[float]], temperature: float, seed: int) -> list[int]:
    """Reproduce the seeded temperature-sampled id sequence.

    Args:
        logits: float64 array of shape (T, V), one logit row per decode step.
        temperature: positive float tau.
        seed: integer seed for np.random.default_rng.

    Returns:
        int64 array of shape (T,) with the sampled token id at each step.
    """
    raise NotImplementedError('your code here')
