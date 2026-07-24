import numpy as np

def _ref_route(weights):
    """Reference implementation of the routing algorithm."""
    if weights.ndim == 2 and weights.shape[1] % 4 == 0:
        groups = weights.reshape(weights.shape[0], -1, 4)
        nonzeros = np.count_nonzero(groups, axis=2)
        if np.all(nonzeros == 2):
            return "tensor-core"
    sparsity = np.mean(weights == 0)
    if sparsity >= 0.5:
        return "csr"
    return "dense"

def _make_cases():
    rng = np.random.default_rng(42)
    cases = []

    # 1) fully dense
    cases.append((np.ones((8, 16)), "dense"))

    # 2) unstructured 70 % zeros -> csr
    w = rng.uniform(-1, 1, (8, 16))
    w[w < 0.7] = 0.0
    cases.append((w, _ref_route(w)))

    # 3) valid 2:4 structured
    w = np.zeros((4, 16))
    for row in range(4):
        for start in range(0, 16, 4):
            cols = rng.choice(4, 2, replace=False)
            w[row, start + cols] = rng.uniform(-1, 1, 2)
    cases.append((w, "tensor-core"))

    # 4) columns not multiple of 4, high sparsity -> csr
    w = np.zeros((4, 9))
    mask = rng.random(w.shape) > 0.4
    w[mask] = rng.uniform(-1, 1, mask.sum())
    cases.append((w, _ref_route(w)))

    # 5) moderate sparsity (30 %) -> dense
    w = rng.uniform(-1, 1, (8, 8))
    w[w < 0.3] = 0.0
    cases.append((w, _ref_route(w)))

    # 6) multiple-of-4 columns but groups not all have 2 non-zeros,
    #    and sparsity 0.625 >= 0.5 -> csr
    w = np.zeros((1, 8))
    w[0, 0] = 1.0
    w[0, 1] = 2.0
    w[0, 2] = 3.0          # first block: 3 non‑zero, invalid
    # second block empty → sparsity = 5/8 = 0.625
    cases.append((w, "csr"))

    # 7) all zeros -> csr
    cases.append((np.zeros((4, 8)), "csr"))

    return cases

def grade(sol, fx) -> dict:
    # fx is unused (no fixtures)
    for w, expected in _make_cases():
        try:
            result = sol.route_tensor(w)
        except Exception:
            return {"exact_match": 0.0}
        if result != expected:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
