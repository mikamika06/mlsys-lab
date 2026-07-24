import numpy as np


def _ref_accumulate_rne(start: float, c: float, n_steps: int, q: float) -> float:
    a = float(start)
    for _ in range(n_steps):
        a = q * round((a + c) / q)
    return a


def _ref_accumulate_stochastic(start: float, c: float, n_steps: int, q: float,
                                rng: np.random.Generator) -> float:
    a = float(start)
    for _ in range(n_steps):
        v = a + c
        lo = np.floor(v / q) * q
        t = (v - lo) / q
        if rng.random() < t:
            a = lo + q
        else:
            a = lo
    return a


_SCENARIOS = [
    dict(start=1000.0, c=0.0003, q=0.01, n_steps=3000, seed_base=0),
    dict(start=1_000_000.0, c=0.0007, q=0.02, n_steps=5000, seed_base=1000),
]
_K = 200


def grade(sol, fx) -> dict:
    worst = 0.0
    for sc in _SCENARIOS:
        start, c, q, n_steps, seed_base = sc["start"], sc["c"], sc["q"], sc["n_steps"], sc["seed_base"]
        exact = start + n_steps * c

        ref_rne = _ref_accumulate_rne(start, c, n_steps, q)
        try:
            got_rne = float(sol.accumulate_rne(start, c, n_steps, q))
        except Exception:
            return {"rel_err": float("inf")}
        if not np.isfinite(got_rne) or abs(got_rne - ref_rne) > 1e-9 * max(1.0, abs(ref_rne)):
            return {"rel_err": float("inf")}

        results = []
        for k in range(_K):
            rng = np.random.default_rng(seed_base + k)
            try:
                v = sol.accumulate_stochastic(start, c, n_steps, q, rng)
            except Exception:
                return {"rel_err": float("inf")}
            try:
                v = float(v)
            except Exception:
                return {"rel_err": float("inf")}
            if not np.isfinite(v):
                return {"rel_err": float("inf")}
            results.append(v)

        mean = float(np.mean(results))
        rel = abs(mean - exact) / (abs(exact) + 1e-300)
        worst = max(worst, rel)

    return {"rel_err": worst}
