import numpy as np


def expected_accepted_tokens(gamma, p):
    g = int(gamma)
    prob = float(p)
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return float(g)

    terms = [(prob ** k) for k in range(1, g + 1)]
    return float(np.sum(terms))


def measure_acceptance_rate(traces):
    if not traces:
        return 0.0
    total_accepted = 0
    total_drafted = 0
    for drafted, accepted in traces:
        total_drafted += int(drafted)
        total_accepted += int(accepted)
    if total_drafted == 0:
        return 0.0
    return float(total_accepted) / float(total_drafted)
