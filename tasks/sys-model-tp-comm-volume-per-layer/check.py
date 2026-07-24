import numpy as np


def _oracle(b, s, h, N):
    elements = np.float64(b) * np.float64(s) * np.float64(h)
    ring_factor = (np.float64(N) - 1.0) / np.float64(N)
    return float(2.0 * elements * ring_factor * 2.0)


def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 1024, 2),
        (8, 2048, 4096, 8),
        (4, 1024, 8192, 16),
        (16, 512, 4096, 64),
        (32, 4096, 12288, 128),
    ]

    ok = 1.0
    for case in cases:
        try:
            got = float(sol.tp_comm_volume_per_layer(*case))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(*case)
        if not np.isclose(got, ref, rtol=0.0, atol=0.0):
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
