import math


def simulate_rope_schedule(stages: int, initial_base: float, target_base: float) -> list[float]:
    if stages <= 1:
        return [float(target_base)]
    bases = []
    for i in range(stages):
        factor = i / (stages - 1)
        val = initial_base * ((target_base / initial_base) ** factor)
        bases.append(float(val))
    return bases
