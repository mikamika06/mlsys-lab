import numpy as np


def _softmax_reference(x):
    z = np.asarray(x, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(123)
    for n, b in [(512, 32), (4096, 64), (32768, 128)]:
        x = rng.normal(size=n).astype(np.float64)
        cases.append((x, b))

    ok = 1.0
    for x, b in cases:
        try:
            got, peak = sol.online_softmax_stream(x, b)
            got = np.asarray(got, dtype=np.float64)
            peak = int(peak)
        except Exception:
            ok = 0.0
            break

        ref = _softmax_reference(x)
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        allowed = 8 * (1 + b)

        if err > 1e-10 or peak > allowed:
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
