import numpy as np
from sampler.core import simulate_execution, statistical_sample, ground_truth_fraction


def compute_bias(period, active_duration, intervals, steps=10000):
    trace = simulate_execution(steps, period, active_duration)
    true_frac = ground_truth_fraction(trace)
    biases = []
    for interval in intervals:
        sample_frac = statistical_sample(trace, interval)
        bias = sample_frac - true_frac
        biases.append({
            "interval": interval,
            "sampled": sample_frac,
            "true": true_frac,
            "bias": bias
        })
    return biases
