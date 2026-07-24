def grade(sol, fx) -> dict:
    import numpy as np

    def ref_flops(S, d, V):
        return int(2 * S * d * V)

    cases = [
        (1, 768, 50257),
        (10, 1024, 30522),
        (5, 2048, 32000),
        (32, 1280, 45000),
        (3, 512, 10000)
    ]

    ok = 1.0
    for S, d, V in cases:
        try:
            got = sol.lm_head_flops(S, d, V)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, int):
            ok = 0.0
            break
        if got != ref_flops(S, d, V):
            ok = 0.0
            break

    return {"exact_match": ok}
