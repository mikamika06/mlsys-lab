import math
import numpy as np

def logsumexp(x: np.ndarray) -> float:
    """Stable computation of the log‑sum‑exp of a 1‑D array."""
    m = float(x[0])
    for i in range(1, len(x)):
        val = float(x[i])
        if val > m:
            m = val

    total = 0.0
    for i in range(len(x)):
        total += math.exp(float(x[i]) - m)

    return float(m + math.log(total))
