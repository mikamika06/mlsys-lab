import random
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = random.Random(42)
    err_max = 0.0
    for _ in range(3):
        n_kv = rng.randint(1, 5)
        g = rng.randint(2, 4)
        d = rng.randint(4, 8)
        n_q = n_kv * g

        Q = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_q)]
        K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_kv)]
        V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_kv)]

        # Reference implementation in pure Python
        O_ref = []
        for i in range(n_q):
            j = i // g
            score = sum(Q[i][k] * K[j][k] for k in range(d))
            row = [score * V[j][k] for k in range(d)]
            O_ref.append(row)

        try:
            O_sol = sol.gqa_attention(Q, K, V, g)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = max_abs_err(O_ref, O_sol)
        if err > err_max:
            err_max = err
    return {"max_abs_err": err_max}
