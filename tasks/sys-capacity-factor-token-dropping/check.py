import math
import numpy as np

def _ref(assignments, num_experts, capacity_factor):
    """Oracle reference: compute the kept/dropped mask."""
    n = len(assignments)
    capacity = math.ceil(capacity_factor * n / num_experts)
    mask = np.zeros(n, dtype=bool)
    for j in range(num_experts):
        indices = np.where(assignments == j)[0]  # ascending by definition
        keep = min(len(indices), capacity)
        mask[indices[:keep]] = True
    return mask

def grade(sol, fx) -> dict:
    cases = [
        # (assignments, num_experts, capacity_factor, description)
        (np.array([0, 1, 0, 1, 0]),            2, 0.8,  "basic overflow"),
        (np.array([0, 0, 0, 1, 1]),            2, 1.0,  "uniform fit"),
        (np.array([0, 0, 0, 0, 0]),            2, 0.4,  "single-expert overload"),
        (np.array([0, 1, 2, 3, 4]),            5, 0.5,  "one token each"),
        (np.array([0, 0, 1, 1, 2, 2, 3, 3]),   4, 0.5,  "pairs with tight cap"),
        (np.array([0, 0, 0, 0, 1, 1, 1, 1]),   2, 0.3,  "heavy skew"),
        (np.array([0]*10 + [1]*10),             2, 0.5,  "large imbalance"),
        (np.array([0, 1, 2]*4),                 3, 1.0,  "perfectly balanced"),
        (np.array([0]*20),                      4, 0.25, "all-to-one"),
        (np.array([3, 1, 3, 1, 3, 1]),         4, 0.5,  "sparse experts"),
    ]

    ok = 1.0
    for assignments, num_experts, cap, _desc in cases:
        try:
            got = np.asarray(
                sol.token_drop_mask(assignments.copy(), num_experts, cap),
                dtype=bool,
            )
        except Exception:
            ok = 0.0
            break
        expected = _ref(assignments, num_experts, cap)
        if not np.array_equal(got, expected):
            ok = 0.0
            break

    return {"exact_match": ok}
