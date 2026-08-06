import numpy as np


def compute_population_p95(cohort_samples: list, cohort_weights: list) -> float:
    all_values = []
    all_weights = []

    for samples, weight in zip(cohort_samples, cohort_weights):
        sample_weight = weight / len(samples)
        all_values.extend(samples)
        all_weights.extend([sample_weight] * len(samples))

    values = np.array(all_values)
    weights = np.array(all_weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    target = 0.95 * total_weight

    idx = np.searchsorted(cum_weights, target)
    return float(values[min(idx, len(values) - 1)])
