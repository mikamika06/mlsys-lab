import dis

import numpy as np

_INSTR_BUDGET = 14


def _ref_clamp(x, lo, hi):
    return max(lo, min(x, hi))


def _build_cases():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(200):
        lo = float(rng.uniform(-50, 50))
        hi = lo + float(rng.uniform(0.0, 50.0))
        x = float(rng.uniform(-100, 100))
        cases.append((x, lo, hi))
    # boundary / edge cases
    cases += [
        (5.0, 5.0, 5.0),
        (0.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
        (-3.0, 0.0, 10.0),
        (15.0, 0.0, 10.0),
        (3, 0, 10),          # ints
        (-1000000.0, -1.0, 1.0),
        (1000000.0, -1.0, 1.0),
    ]
    return cases


def grade(sol, fx) -> dict:
    cases = _build_cases()

    exact = 1.0
    for x, lo, hi in cases:
        ref = _ref_clamp(x, lo, hi)
        try:
            got = sol.clamp(x, lo, hi)
        except Exception:
            exact = 0.0
            break
        if got != ref:
            exact = 0.0
            break

    try:
        n_instr = len(list(dis.get_instructions(sol.clamp)))
    except Exception:
        n_instr = float("inf")

    instr_budget_ok = 1.0 if n_instr <= _INSTR_BUDGET else 0.0

    return {
        "exact_match": exact,
        "instr_budget_ok": instr_budget_ok,
    }
