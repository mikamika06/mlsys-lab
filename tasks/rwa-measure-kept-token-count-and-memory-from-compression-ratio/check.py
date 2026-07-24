import numpy as np

def grade(sol, fx) -> dict:
    """
    Grade a candidate implementation of measure_kept_tokens_and_memory.
    The grader uses NumPy to compute the reference values and compares
    against the student's output.  It returns a dictionary with the key
    'exact_match' which is 1.0 on success and 0.0 otherwise.
    """
    # Prepare test cases: (compression_ratio, seq_len, full_bytes)
    tests = [
        (0.25, 100, 8000),
        (0.5, 200, 16000),
        (0.123456, 987, 12345),
        (1.0, 50, 5000),   # all memory discarded
        (0.0, 30, 3000),   # no compression
    ]

    ok = 1.0
    for r, n, B in tests:
        try:
            kept, saved = sol.measure_kept_tokens_and_memory(r, n, B)
        except Exception:
            return {"exact_match": 0.0}

        ref_kept = int(np.round((1 - r) * n))
        ref_saved = (r * B)

        if kept != ref_kept:
            ok = 0.0
            break

        # relative error for the float value
        denom = abs(ref_saved) + 1e-12
        rel_err = abs(saved - ref_saved) / denom
        if rel_err > 1e-9:
            ok = 0.0
            break

    return {"exact_match": ok}
