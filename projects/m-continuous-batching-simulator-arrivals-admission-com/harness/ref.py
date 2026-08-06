import random
from cbsim.simulator import Request

def get_test_workload():
    rng = random.Random(42)
    reqs = []
    for i in range(20):
        arr = rng.randint(0, 10) * 2
        p_len = rng.randint(8, 32)
        g_len = rng.randint(8, 32)
        reqs.append(Request(i, arr, p_len, g_len))
    return reqs
