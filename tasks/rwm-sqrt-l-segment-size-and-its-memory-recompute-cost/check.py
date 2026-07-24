import math


def _oracle(L):
    segment_size = int(round(math.sqrt(L)))
    stored_activations = 2 * segment_size
    extra_forward = L
    return (segment_size, stored_activations, extra_forward)


def grade(sol, fx) -> dict:
    cases = [1, 2, 3, 4, 7, 16, 31, 64, 100, 257, 1024]
    ok = 1.0
    for L in cases:
        try:
            got = tuple(sol.checkpoint_cost(L))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(L):
            ok = 0.0
            break
    return {"exact_match": ok}
