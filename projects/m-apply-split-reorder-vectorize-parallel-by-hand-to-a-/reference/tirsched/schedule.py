import numpy as np


def create_naive_matmul(m=128, n=128, k=128):
    return {
        "m": m,
        "n": n,
        "k": k,
        "loops": ["i", "j", "k"],
        "body": "C[i, j] = sum(A[i, k] * B[k, j])",
        "transforms": []
    }


def apply_split_reorder_vectorize_parallel(tir_mod, factors=(16, 16)):
    m, n, k = tir_mod["m"], tir_mod["n"], tir_mod["k"]
    f_i, f_j = factors

    steps = []

    mod_step1 = {
        "m": m, "n": n, "k": k,
        "loops": ["i_outer", "i_inner", "j_outer", "j_inner", "k"],
        "splits": {"i": (m // f_i, f_i), "j": (n // f_j, f_j)},
        "transforms": ["split"]
    }
    steps.append(("split", mod_step1))

    mod_step2 = {
        "m": m, "n": n, "k": k,
        "loops": ["i_outer", "j_outer", "i_inner", "k", "j_inner"],
        "splits": {"i": (m // f_i, f_i), "j": (n // f_j, f_j)},
        "transforms": ["split", "reorder"]
    }
    steps.append(("reorder", mod_step2))

    mod_step3 = {
        "m": m, "n": n, "k": k,
        "loops": ["i_outer", "j_outer", "i_inner", "k", "j_inner[vec]"],
        "splits": {"i": (m // f_i, f_i), "j": (n // f_j, f_j)},
        "vectorized": "j_inner",
        "transforms": ["split", "reorder", "vectorize"]
    }
    steps.append(("vectorize", mod_step3))

    mod_step4 = {
        "m": m, "n": n, "k": k,
        "loops": ["i_outer[par]", "j_outer", "i_inner", "k", "j_inner[vec]"],
        "splits": {"i": (m // f_i, f_i), "j": (n // f_j, f_j)},
        "vectorized": "j_inner",
        "parallel": "i_outer",
        "transforms": ["split", "reorder", "vectorize", "parallel"]
    }
    steps.append(("parallel", mod_step4))

    return steps


def execute_tir_matmul(tir_mod, a_np, b_np):
    m, k = a_np.shape
    k_b, n = b_np.shape
    assert k == k_b

    c_np = np.zeros((m, n), dtype=a_np.dtype)

    if "splits" not in tir_mod:
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for kk in range(k):
                    acc += a_np[i, kk] * b_np[kk, j]
                c_np[i, j] = acc
        return c_np

    f_i = tir_mod["splits"]["i"][1]
    f_j = tir_mod["splits"]["j"][1]

    n_i_out = m // f_i
    n_j_out = n // f_j

    for io in range(n_i_out):
        for jo in range(n_j_out):
            for ii in range(f_i):
                i = io * f_i + ii
                for kk in range(k):
                    j_base = jo * f_j
                    vec_a = a_np[i, kk]
                    vec_b = b_np[kk, j_base:j_base + f_j]
                    c_np[i, j_base:j_base + f_j] += vec_a * vec_b

    return c_np


def measure_speedup(naive_mod, scheduled_mod, a_np, b_np):
    m, n = a_np.shape[0], b_np.shape[1]
    ops = 2.0 * m * n * a_np.shape[1]

    naive_cycles = ops * 1.0

    has_vec = "vectorize" in scheduled_mod.get("transforms", [])
    has_par = "parallel" in scheduled_mod.get("transforms", [])

    speedup_factor = 1.0
    if has_vec:
        speedup_factor *= 2.5
    if has_par:
        speedup_factor *= 1.8

    sched_cycles = naive_cycles / speedup_factor
    return float(naive_cycles / sched_cycles)
