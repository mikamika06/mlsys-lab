import numpy as np


def generate_fixtures():
    np.random.seed(1337)
    fixtures = []
    for i in range(3):
        w = np.random.randn(12, 12).tolist()
        h = np.eye(12).tolist()
        sparsity = 0.25 + 0.1 * i
        q_bits = 4
        fixtures.append({
            "weights": w,
            "hessian": h,
            "sparsity": sparsity,
            "q_bits": q_bits
        })
    return fixtures
