import numpy as np

def _oracle(block_mask):
    """NumPy oracle: count skipped blocks and compute dense-FLOP fraction."""
    total = block_mask.size
    if total == 0:
        return 0, 1.0
    computed = int(np.sum(block_mask))
    skipped = total - computed
    flop_fraction = computed / total
    return skipped, float(flop_fraction)

def grade(sol, fx) -> dict:
    test_cases = [
        np.ones((4, 4), dtype=bool),                               # all True
        np.zeros((3, 5), dtype=bool),                               # all False
        np.array([[True, False, True], [False, False, True]]),      # mixed
        np.array([[False]]),                                         # single False
        np.array([[True]]),                                          # single True
    ]
    # Seeded random case (deterministic across machines)
    rng = np.random.RandomState(42)
    test_cases.append(rng.random((8, 12)) > 0.5)

    ok_exact = 1.0
    ok_frac = 1.0

    for bm in test_cases:
        try:
            result = sol.measure_block_sparsity(bm)
            skipped = int(result[0])
            frac = float(result[1])
        except Exception:
            ok_exact = 0.0
            ok_frac = 0.0
            break

        ref_skipped, ref_frac = _oracle(bm)

        # Integer comparison for skipped count
        if skipped != ref_skipped:
            ok_exact = 0.0

        # Relative-error comparison for fraction
        if ref_frac > 0:
            rel = abs(frac - ref_frac) / ref_frac
        elif frac > 0:
            rel = float("inf")
        else:
            rel = 0.0
        if rel > 1e-9:
            ok_frac = 0.0

    return {"exact_match": ok_exact, "flop_fraction_accuracy": ok_frac}
