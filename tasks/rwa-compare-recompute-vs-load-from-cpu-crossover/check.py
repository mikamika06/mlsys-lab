import numpy as np


def _oracle(recompute_coeff, kv_bytes_per_token, bandwidth_bytes_per_s):
    recompute_coeff = float(recompute_coeff)
    transfer_coeff = float(kv_bytes_per_token) / float(bandwidth_bytes_per_s)

    limit = 100000
    lengths = np.arange(1, limit + 1, dtype=np.float64)
    recompute = recompute_coeff * lengths * lengths
    load = transfer_coeff * lengths
    matches = np.nonzero(load <= recompute)[0]
    if len(matches) == 0:
        raise RuntimeError("oracle range too small")
    return int(matches[0] + 1)


def grade(sol, fx) -> dict:
    cases = [
        (2.0, 1000.0, 1000.0),
        (0.5, 4096.0, 1024.0),
        (3.25, 1_000_000.0, 50_000_000.0),
        (1e-3, 512.0, 4096.0),
        (7.5, 16384.0, 100_000.0),
    ]

    ok = 1.0
    for args in cases:
        try:
            got = sol.crossover_seq_len(*args)
        except Exception:
            ok = 0.0
            break

        ref = _oracle(*args)
        if not isinstance(got, (int, np.integer)) or int(got) != ref:
            ok = 0.0
            break

    return {"argmin_index": ok}
