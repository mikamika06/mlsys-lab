import numpy as np


def _oracle(T, k, d, N, bytes_per_elem):
    values = np.asarray([T, k, d, bytes_per_elem], dtype=np.int64)
    phase = np.prod(values, dtype=np.int64)
    total = np.multiply(np.int64(2), phase, dtype=np.int64)
    return int(total)


def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 1, 1, 2),
        (1024, 2, 4096, 8, 2),
        (8192, 4, 8192, 16, 2),
        (512, 2, 1024, 4, 4),
        (37, 3, 768, 8, 2),
    ]

    ok = 1.0
    for case in cases:
        try:
            got = sol.moe_ep_comm_bytes(*case)
        except Exception:
            ok = 0.0
            break

        expected = _oracle(*case)
        if got != expected or not isinstance(got, int):
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
