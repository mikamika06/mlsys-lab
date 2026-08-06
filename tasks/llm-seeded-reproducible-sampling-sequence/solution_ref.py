import math
import random


def sample_sequence(logits: list[list[float]], temperature: float, seed: int) -> list[int]:
    """Reproduce a seeded temperature-sampled id sequence via inverse-CDF draws.

    A random.Random(seed) is created, and one uniform is consumed per
    decode step, in order. Returns a list of integers of length T.
    """
    T = len(logits)
    V = len(logits[0])
    rng = random.Random(seed)
    ids = [0] * T
    for t in range(T):
        row = logits[t]
        z = [val / temperature for val in row]

        max_z = z[0]
        for i in range(1, V):
            if z[i] > max_z:
                max_z = z[i]

        z_shifted = [val - max_z for val in z]

        e = [0.0] * V
        for i in range(V):
            e[i] = math.exp(z_shifted[i])

        sum_e = 0.0
        for i in range(V):
            sum_e += e[i]

        p = [0.0] * V
        for i in range(V):
            p[i] = e[i] / sum_e

        cdf = [0.0] * V
        acc = 0.0
        for i in range(V):
            acc += p[i]
            cdf[i] = acc

        u = rng.random()

        idx = V
        for i in range(V):
            if cdf[i] > u:
                idx = i
                break

        if idx >= V:
            idx = V - 1
        ids[t] = idx
    return ids
