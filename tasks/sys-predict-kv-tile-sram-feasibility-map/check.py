import numpy as np


def _oracle(configs):
    arr = np.asarray(configs, dtype=np.int64)
    seq = arr[:, 0]
    d = arr[:, 1]
    sram = arr[:, 2]
    required = np.multiply(np.multiply(seq, d), 4, dtype=np.int64)
    return list(required <= sram)


def grade(sol, fx) -> dict:
    cases = [
        [(64, 64, 16384), (128, 128, 65536), (256, 128, 65536)],
        [(1, 1, 3), (1, 1, 4), (32, 80, 10240), (32, 80, 10239)],
        [(128, 64, 32768), (256, 64, 65536), (512, 64, 131071)],
        [(16, 256, 16384), (16, 256, 16385), (17, 256, 17408)],
    ]
    ok = 1.0
    for configs in cases:
        try:
            got = list(sol.kv_tile_sram_feasibility_map(configs))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(configs)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
