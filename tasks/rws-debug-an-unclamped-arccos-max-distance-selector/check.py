import numpy as np


def _oracle(query, candidates):
    q = np.asarray(query, dtype=np.float64)
    C = np.asarray(candidates, dtype=np.float64)
    dots = C @ q
    denom = np.linalg.norm(q) * np.linalg.norm(C, axis=1)
    cos = dots / denom
    cos = np.clip(cos, -1.0, 1.0)
    dist = np.arccos(cos)
    return int(np.argmin(dist))


def _cases():
    cases = []

    # Fixed case: candidate 0 is an (almost) exact positive scalar multiple
    # of the query. In float64, dot(q, c) / (|q| |c|) evaluates to a value
    # a few ULPs above 1.0 for these specific numbers, so arccos WITHOUT
    # clipping returns nan for this candidate.
    q = np.array([
        3.925523255328582, 14.473776284320387, -0.15229056804655983,
        11.402525936516732, 15.347236382320284, 12.588110005895928,
        -25.88731990747355, 13.447459508147771, 3.7170072637092675,
        4.638010595520001, 4.06293791502263, 4.189126407186102,
        3.4958628701902366, -3.928164838578263, -20.812649565141637,
    ])
    c0 = np.array([
        1433.2464341558157, 5284.515438830284, -55.602756476352965,
        4163.172289629925, 5603.4241522448365, 4596.040477972231,
        -9451.710392206469, 4909.797276667577, 1357.1152327810246,
        1693.3824398015233, 1483.4178528934035, 1529.490489497783,
        1276.3732799688837, -1434.2109016998538, -7598.899263705159,
    ])
    rng = np.random.default_rng(123)
    others = rng.standard_normal((3, 15))
    candidates = np.vstack([c0[None, :], others])
    cases.append((q, candidates))

    # Random well-separated cases (no NaN edge, just argmin-vs-argmax).
    rng = np.random.default_rng(7)
    for _ in range(6):
        d = int(rng.integers(3, 12))
        k = int(rng.integers(3, 8))
        q = rng.standard_normal(d)
        C = rng.standard_normal((k, d))
        cases.append((q, C))

    return cases


def grade(sol, fx) -> dict:
    cases = _cases()
    correct = 0
    for q, C in cases:
        ref = _oracle(q, C)
        try:
            got = sol.select_min_angle_block(q.copy(), C.copy())
            got = int(got)
        except Exception:
            got = None
        if got == ref:
            correct += 1
    return {"argmin_index": correct / len(cases)}
