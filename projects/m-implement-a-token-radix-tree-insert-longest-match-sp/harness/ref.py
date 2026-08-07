import random

def get_test_traces():
    rng = random.Random(42)
    base = [rng.randint(0, 100) for _ in range(50)]
    traces = []
    for _ in range(10):
        branch_point = rng.randint(10, len(base))
        branch = base[:branch_point] + [rng.randint(101, 200) for _ in range(30)]
        traces.append(branch)
    return traces
