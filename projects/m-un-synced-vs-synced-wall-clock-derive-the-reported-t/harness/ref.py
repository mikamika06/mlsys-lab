import random


def get_test_cases_m1():
    rng = random.Random(1337)
    cases = []
    for _ in range(10):
        launches = [rng.uniform(0.1, 1.0) for _ in range(4)]
        kernels = [rng.uniform(2.0, 5.0) for _ in range(4)]
        from timing.derive import derive_reported_time_gap as ref_fn
        ans = ref_fn(launches, kernels)
        cases.append((launches, kernels, ans))
    return cases


def get_test_cases_m2():
    rng = random.Random(1337)
    cases = []
    for i in range(5):
        events = [
            {"type": "kernel", "duration": 2.0},
            {"type": "gpu_idle", "duration": 150.0 + i * 20, "threshold": 50.0, "has_sync": False}
        ]
        from timing.trace import find_missing_cuda_synchronize as ref_fn
        ans = ref_fn(events)
        cases.append((events, ans))
    return cases
