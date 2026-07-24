import numpy as np


def _stage_bytes(psi, n_devices, stage):
    if stage == 0:
        return 16.0 * psi
    if stage == 1:
        return 4.0 * psi + 12.0 * psi / n_devices
    if stage == 2:
        return 2.0 * psi + 14.0 * psi / n_devices
    if stage == 3:
        return 16.0 * psi / n_devices
    raise ValueError(stage)


def _oracle(psi, n_devices, budget_bytes):
    for stage in (0, 1, 2, 3):
        if _stage_bytes(psi, n_devices, stage) <= budget_bytes:
            return stage
    return -1


def _gen_case(rng):
    psi = float(rng.integers(1, 20)) * 10 ** int(rng.integers(6, 11))  # ~1e6..~2e10 params
    n_devices = int(rng.integers(1, 65))
    target_stage = int(rng.choice([0, 1, 2, 3, -1]))
    if target_stage == -1:
        # budget below even the best (stage 3) requirement
        budget = _stage_bytes(psi, n_devices, 3) * float(rng.uniform(0.1, 0.99))
    else:
        exact = _stage_bytes(psi, n_devices, target_stage)
        # perturb budget upward a bit so it's strictly >= this stage's need,
        # but stays below the next-cheaper stage's need (if any).
        if target_stage == 0:
            budget = exact * float(rng.uniform(1.0, 3.0))
        else:
            cheaper = _stage_bytes(psi, n_devices, target_stage - 1)
            hi = max(exact, min(cheaper, exact * 1.2)) if cheaper > exact else exact * 1.2
            lo = exact
            budget = float(rng.uniform(lo, max(lo, hi)))
    return psi, n_devices, budget


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [_gen_case(rng) for _ in range(14)]
    cases.append((1e9, 8, 16e9))     # huge budget -> stage 0 fits
    cases.append((1e9, 8, 1.0))      # tiny budget -> nothing fits

    ok = 1.0
    for psi, n_devices, budget in cases:
        expected = _oracle(psi, n_devices, budget)
        try:
            got = int(sol.min_zero_stage(psi, n_devices, budget))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
