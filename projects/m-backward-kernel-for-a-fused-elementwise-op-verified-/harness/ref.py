import numpy as np


def generate_test_cases():
    rng = np.random.default_rng(12345)
    cases = []

    x1 = rng.standard_normal(8)
    map1 = np.array([7, 2, 0, 4, 1, 5, 3, 6], dtype=np.int64)
    gout1 = rng.standard_normal(len(map1))
    cases.append({"x": x1, "index_map": map1, "grad_output": gout1, "overlaps": False})

    x2 = rng.standard_normal(10)
    map2 = np.array([0, 3, 2, 0, 7, 3, 9, 2, 0, 1, 5, 3], dtype=np.int64)
    gout2 = rng.standard_normal(len(map2))
    cases.append({"x": x2, "index_map": map2, "grad_output": gout2, "overlaps": True})

    x3 = rng.standard_normal(5)
    map3 = np.array([2, 2, 2, 2], dtype=np.int64)
    gout3 = rng.standard_normal(len(map3))
    cases.append({"x": x3, "index_map": map3, "grad_output": gout3, "overlaps": True})

    return cases
