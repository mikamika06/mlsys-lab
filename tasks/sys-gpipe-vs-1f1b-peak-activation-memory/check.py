import random
from mlsys import scorers

def _ref_gpipe(M, a_bytes, d):
    return M * a_bytes * d

def _ref_1f1b(p, a_bytes, d):
    return (p - 1) * a_bytes * d

def grade(sol, fx) -> dict:
    rng = random.Random(42)
    cases = []
    # gpipe cases
    for _ in range(6):
        M = rng.randint(2, 32)
        d = rng.randint(2, 8)
        a_bytes = rng.choice([512, 1024, 2048, 4096])
        cases.append(("gpipe", M, d, a_bytes, d, _ref_gpipe(M, a_bytes, d)))
    # 1f1b cases (p == d in 1f1b)
    for _ in range(6):
        p = rng.randint(2, 8)
        d = p
        a_bytes = rng.choice([512, 1024, 2048, 4096])
        M = rng.randint(2, 32)  # not used, but passed
        cases.append(("1f1b", M, p, a_bytes, d, _ref_1f1b(p, a_bytes, d)))

    score = 1.0
    for schedule, M, p, a_bytes, d, expected in cases:
        try:
            got = sol.compute_peak_activation_bytes(schedule, M, p, a_bytes, d)
        except Exception:
            score = 0.0
            break
        if not isinstance(got, int) and not (isinstance(got, float) and got == int(got)):
            score = 0.0
            break
        if int(got) != expected:
            score = 0.0
            break
    return {"size_ratio_pipe": score}
