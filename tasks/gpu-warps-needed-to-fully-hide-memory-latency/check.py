import math

def grade(sol, fx) -> dict:
    test_cases = [
        (200, 10),
        (100, 7),
        (256, 4),
        (512, 32),
        (300, 1),
        (1,   1),
        (128, 16),
        (999, 13),
        (64,  9),
        (400, 20),
    ]
    total = len(test_cases)
    correct = 0
    for L, I in test_cases:
        ref = math.ceil(L / I)
        try:
            got = sol.min_warps_to_hide_latency(L, I)
        except Exception:
            got = None
        if got == ref:
            correct += 1
    exact_match = correct / total
    return {"exact_match": exact_match}
