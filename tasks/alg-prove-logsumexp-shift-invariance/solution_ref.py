import math

def logsumexp(x: list[float]) -> float:
    """Stable computation of the log‑sum‑exp of a list."""
    m = float(x[0])
    for i in range(1, len(x)):
        val = float(x[i])
        if val > m:
            m = val

    total = 0.0
    for i in range(len(x)):
        total += math.exp(float(x[i]) - m)

    return float(m + math.log(total))
