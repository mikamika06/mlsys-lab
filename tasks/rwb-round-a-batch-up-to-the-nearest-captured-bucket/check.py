def _reference(captured_sizes, batch_sizes):
    chosen = []
    padded = []
    max_bucket = captured_sizes[-1] if captured_sizes else None
    for b in batch_sizes:
        bucket = next((c for c in captured_sizes if c >= b), None)
        if bucket is None:
            chosen.append(-1)
            padded.append(0)
        else:
            chosen.append(bucket)
            padded.append(bucket - b)
    return chosen, padded


def grade(sol, fx) -> dict:
    # Test cases: (captured_sizes, batch_sizes)
    cases = [
        ([32, 64, 128], [10, 33, 65, 200]),
        ([16, 32, 48],   [0, 15, 17, 30, 31, 49]),
        ([8, 16, 24, 32], []),
        ([],              [5, 12]),
    ]
    ok = 1.0
    for captured, batches in cases:
        try:
            got_chosen, got_padded = sol.round_to_bucket(captured, batches)
        except Exception:
            return {"exact_match": 0.0}
        exp_chosen, exp_padded = _reference(captured, batches)
        if list(got_chosen) != exp_chosen or list(got_padded) != exp_padded:
            ok = 0.0
            break
    return {"exact_match": ok}
