import numpy as np
from tirsched.schedule import create_naive_matmul, apply_split_reorder_vectorize_parallel, execute_tir_matmul
from tirsched.analysis import print_tir_loop_nest


def test_schedule_vectorization_and_correctness():
    m, n, k = 64, 64, 64
    naive = create_naive_matmul(m, n, k)
    steps = apply_split_reorder_vectorize_parallel(naive, factors=(16, 16))

    assert len(steps) == 4
    final_step_name, final_mod = steps[-1]

    assert "vectorize" in final_mod.get("transforms", [])
    assert "j_inner" in final_mod.get("vectorized", "")

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((m, k)).astype("float32")
    b_np = rng.standard_normal((k, n)).astype("float32")

    ref_c = a_np @ b_np
    out_c = execute_tir_matmul(final_mod, a_np, b_np)

    err = np.max(np.abs(out_c - ref_c))
    assert err < 1e-4
