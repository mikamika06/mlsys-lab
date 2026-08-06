import numpy as np
from tirsched.schedule import create_naive_matmul, apply_split_reorder_vectorize_parallel, execute_tir_matmul


def generate_fixtures(seed=1001):
    rng = np.random.default_rng(seed)
    m, n, k = 64, 64, 64
    a_np = rng.standard_normal((m, k)).astype("float32")
    b_np = rng.standard_normal((k, n)).astype("float32")
    c_ref = a_np @ b_np
    return m, n, k, a_np, b_np, c_ref
