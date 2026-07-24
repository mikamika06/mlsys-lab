import math


def _oracle(N: int, d: int, M: int, elem_bytes: int = 4) -> dict:
    naive_elems = 4 * N * d + 4 * N * N

    Bc = max(1, min(N, math.ceil(M / (4 * d))))
    Tc = math.ceil(N / Bc)
    tiled_elems = 2 * N * d + 3 * Tc * N * d

    naive_bytes = naive_elems * elem_bytes
    tiled_bytes = tiled_elems * elem_bytes

    return {
        "naive_bytes": naive_bytes,
        "tiled_bytes": tiled_bytes,
        "size_ratio": tiled_bytes / naive_bytes,
    }


def grade(sol, fx) -> dict:
    configs = [
        (256, 32, 20000),
        (1024, 64, 100000),
        (2048, 32, 50000),
        (4096, 64, 400000),  # the "given config" -- ratio must be < 0.1
    ]

    worst_rel_err = 0.0
    hint_size_ratio = float("inf")

    for N, d, M in configs:
        ref = _oracle(N, d, M)
        try:
            got = sol.hbm_traffic(N, d, M)
        except Exception:
            return {"rel_err": float("inf"), "size_ratio": float("inf")}

        try:
            got_ratio = float(got["size_ratio"])
        except Exception:
            return {"rel_err": float("inf"), "size_ratio": float("inf")}

        ref_ratio = ref["size_ratio"]
        rel_err = abs(got_ratio - ref_ratio) / (abs(ref_ratio) + 1e-12)
        worst_rel_err = max(worst_rel_err, rel_err)

        if (N, d, M) == (4096, 64, 400000):
            hint_size_ratio = got_ratio

    return {"rel_err": worst_rel_err, "size_ratio": hint_size_ratio}
