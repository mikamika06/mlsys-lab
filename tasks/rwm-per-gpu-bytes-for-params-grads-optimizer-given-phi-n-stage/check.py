import random

def _oracle(phi, n_gpus, stage):
    if stage == 0:
        return 16 * phi
    elif stage == 1:
        return 4 * phi + (12 * phi) // n_gpus
    elif stage == 2:
        return (4 * phi) // n_gpus + 12 * phi
    else:
        raise ValueError("stage must be 0, 1 or 2")

def grade(sol, fx):
    ok = 1.0
    for _ in range(20):
        phi = random.randint(10_000, 100_000_000)
        n_gpus = random.choice([2, 4, 8, 16])
        stage = random.randint(0, 2)
        try:
            got = sol.per_gpu_bytes(phi, n_gpus, stage)
        except Exception:
            ok = 0.0
            break
        ref = _oracle(phi, n_gpus, stage)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
