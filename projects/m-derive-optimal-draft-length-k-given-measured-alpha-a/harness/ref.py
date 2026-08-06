import random

def get_test_cases():
    cases = []
    rng = random.Random(42)
    for _ in range(10):
        alpha = round(rng.uniform(0.3, 0.9), 2)
        cost_ratio = round(rng.uniform(0.1, 0.5), 2)
        cases.append({"alpha": alpha, "cost_ratio": cost_ratio})
    return cases

def get_traces():
    traces = []
    rng = random.Random(43)
    for _ in range(5):
        trace = {
            "depth": rng.randint(3, 6),
            "branch_factor": rng.randint(2, 4),
            "paths": [rng.choices([0, 1], weights=[1-p, p], k=rng.randint(2, 5)) for p in [0.7, 0.6, 0.5]]
        }
        traces.append(trace)
    return traces
