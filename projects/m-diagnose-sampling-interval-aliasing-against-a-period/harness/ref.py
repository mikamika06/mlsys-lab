import numpy as np


def simulate_execution(steps, period, active_duration):
    trace = np.zeros(steps, dtype=bool)
    for i in range(0, steps, period):
        end = min(i + active_duration, steps)
        trace[i:end] = True
    return trace


def statistical_sample(execution_trace, interval, offset=0):
    if len(execution_trace) == 0:
        return 0.0
    indices = np.arange(offset, len(execution_trace), interval)
    if len(indices) == 0:
        return 0.0
    return float(np.mean(execution_trace[indices]))


def ground_truth_fraction(execution_trace):
    if len(execution_trace) == 0:
        return 0.0
    return float(np.mean(execution_trace))


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
