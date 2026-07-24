import numpy as np


def _oracle(params, world_size, bytes_per_param):
    p = np.float64(params)
    w = np.float64(world_size)
    b = np.float64(bytes_per_param)
    dp = (2.0 * (w - 1.0) / w) * p * b
    zero3 = ((w - 1.0) / w) * p * b + ((w - 1.0) / w) * p * b
    ratio = zero3 / dp
    return {
        "dp_bytes": float(dp),
        "zero3_bytes": float(zero3),
        "ratio": float(ratio),
    }


def grade(sol, fx) -> dict:
    cases = [
        (1000, 2, 4),
        (1000000, 8, 2),
        (7654321, 16, 2),
        (4096, 4, 8),
        (99999999, 64, 2),
    ]
    ok = 1.0
    for params, world_size, bytes_per_param in cases:
        try:
            got = sol.compare_zero3_dp_comm(
                params,
                world_size,
                bytes_per_param,
            )
        except Exception:
            ok = 0.0
            break
        ref = _oracle(params, world_size, bytes_per_param)
        if set(got.keys()) != set(ref.keys()):
            ok = 0.0
            break
        for key in ref:
            if float(got[key]) != ref[key]:
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"modeled_mem_access": ok}
