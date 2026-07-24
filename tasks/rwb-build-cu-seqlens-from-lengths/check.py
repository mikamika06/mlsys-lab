import numpy as np

def grade(sol, fx) -> dict:
    # Test cases: varying lengths and sizes
    rng = np.random.default_rng(42)
    cases = [
        np.array([3, 2, 4], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.arange(10, dtype=np.int32),
        rng.integers(1, 5, size=20, dtype=np.int32),
        rng.integers(0, 100, size=50, dtype=np.int32)
    ]

    ok = 1.0
    for lengths in cases:
        try:
            got = sol.build_cu_seqlens(lengths)
            # Convert to list for element‑wise comparison
            got_list = list(got.tolist())
            expected = np.concatenate([[0], np.cumsum(lengths)]).tolist()
            if got_list != expected:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
