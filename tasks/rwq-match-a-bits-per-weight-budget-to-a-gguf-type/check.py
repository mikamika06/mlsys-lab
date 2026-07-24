import numpy as np

# GGUF type definitions: (block_bytes, weights_per_block)
GGUF_TYPES = [
    ("Q2_K",   16, 64),
    ("Q4_K_M", 32, 64),
    ("Q6_K",   48, 64),
    ("Q8_0",   64, 64),
]

def _compute_bpw():
    """Return a NumPy array of bpw values for the four types."""
    block_bytes = np.array([t[1] for t in GGUF_TYPES], dtype=np.float64)
    weights     = np.array([t[2] for t in GGUF_TYPES], dtype=np.float64)
    return 8.0 * block_bytes / weights

def _oracle_index(target):
    """Oracle: index of type with bpw closest to target."""
    bpw = _compute_bpw()
    diff = np.abs(bpw - target)
    return int(np.argmin(diff))

def grade(sol, fx) -> dict:
    # Test cases: (target_bpw, expected_index)
    tests = [
        (5.3, 2),
        (7.9, 3),
        (1.8, 0),
        (4.0, 1),   # exact match
        (6.1, 2),
    ]
    ok = 1.0
    for target, expected in tests:
        try:
            got = sol.match_bpw(float(target))
        except Exception:
            return {"argmin_index": 0.0}
        if got != expected:
            ok = 0.0
            break
    return {"argmin_index": ok}
