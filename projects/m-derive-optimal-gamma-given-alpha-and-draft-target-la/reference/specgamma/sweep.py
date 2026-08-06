import numpy as np
from specgamma.derivation import optimal_gamma


def sweep_gamma(alpha: float, c: float, max_gamma: int = 8) -> dict:
    gammas = list(range(1, max_gamma + 1))
    throughputs = []
    for g in gammas:
        exp_acc = (1.0 - alpha ** (g + 1)) / (1.0 - alpha)
        tput = exp_acc / (1.0 + c * g)
        throughputs.append(float(tput))
    opt = optimal_gamma(alpha, c, max_gamma)
    return {"gammas": gammas, "throughputs": throughputs, "optimal": opt}
