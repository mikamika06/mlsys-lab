import numpy as np


def simulate_stability(losses, threshold=10.0):
    stable = True
    diverge_step = -1
    for i, l in enumerate(losses):
        if l > threshold or np.isnan(l) or np.isinf(l):
            stable = False
            diverge_step = i
            break
    return {"stable": stable, "diverge_step": diverge_step}
