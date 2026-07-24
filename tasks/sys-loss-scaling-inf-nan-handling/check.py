import numpy as np


def _oracle(scaled_grads, scale):
    skip = any(
        not np.all(np.isfinite(np.asarray(g, dtype=np.float32))) for g in scaled_grads
    )
    unscaled = [
        np.asarray(g, dtype=np.float32) / np.float32(scale) for g in scaled_grads
    ]
    return bool(skip), unscaled


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    g1 = [
        (rng.standard_normal((4, 5)) * 100.0).astype(np.float32),
        (rng.standard_normal((3,)) * 10.0).astype(np.float32),
    ]
    cases.append((g1, 1024.0))

    g2 = [
        rng.standard_normal((3, 4)).astype(np.float32),
        rng.standard_normal((2,)).astype(np.float32),
    ]
    g2[1][0] = np.inf
    cases.append((g2, 65536.0))

    g3 = [rng.standard_normal((5,)).astype(np.float32)]
    g3[0][2] = np.nan
    cases.append((g3, 256.0))

    g4 = [rng.standard_normal((2, 2)).astype(np.float32)]
    cases.append((g4, 2.0))

    g5 = [rng.standard_normal((3, 3)).astype(np.float32)]
    g5[0][1, 1] = -np.inf
    cases.append((g5, 512.0))

    g6 = [
        (rng.standard_normal((6,)) * 1000.0).astype(np.float32),
        (rng.standard_normal((2, 2)) * 0.01).astype(np.float32),
    ]
    cases.append((g6, 32768.0))

    return cases


def grade(sol, fx) -> dict:
    ok = 1.0
    for grads, scale in _cases():
        grads_in = [np.array(g, copy=True) for g in grads]
        ref_skip, ref_unscaled = _oracle(grads, scale)
        try:
            got_skip, got_unscaled = sol.unscale_and_check(grads_in, scale)
        except Exception:
            return {"exact_match": 0.0}

        if bool(got_skip) != ref_skip:
            ok = 0.0
            continue
        try:
            got_unscaled = list(got_unscaled)
        except TypeError:
            ok = 0.0
            continue
        if len(got_unscaled) != len(ref_unscaled):
            ok = 0.0
            continue
        for a, b in zip(got_unscaled, ref_unscaled):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                ok = 0.0
                break
    return {"exact_match": ok}
