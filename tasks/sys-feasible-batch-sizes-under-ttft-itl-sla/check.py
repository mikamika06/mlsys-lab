import numpy as np


def _oracle(batch_sizes, prompt_tokens, gen_tokens, sla_ttft_ms, sla_itl_ms):
    b = np.asarray(batch_sizes, dtype=np.float64)
    ttft = 20.0 + 0.05 * prompt_tokens * b
    itl = 5.0 + 0.01 * gen_tokens * b
    return ((ttft <= sla_ttft_ms) & (itl <= sla_itl_ms)).tolist()


def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 4, 8, 16], 100, 50, 30, 15),
        ([1, 2, 4, 8, 16, 32], 32, 400, 100, 15),
        ([1, 5, 10, 20, 40], 256, 20, 200, 20),
        ([2, 4, 8, 16, 32], 64, 300, 100, 20),
        ([1, 3, 7, 15, 31], 512, 128, 1000, 20),
    ]

    ok = 1.0
    for args in cases:
        try:
            got = sol.feasible_batch_sizes(*args)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(*args):
            ok = 0.0
            break

    return {"exact_match": ok}
