import ref

def check(workdir):
    from spec.sweep import find_optimal_draft_n
    out = {"optimal_matched": 0.0}
    def mock_eval(n):
        return float(100 - (n - 4) ** 2)

    n_values = [1, 2, 4, 8, 16]
    want = ref.find_optimal_draft_n(mock_eval, n_values)
    got = find_optimal_draft_n(mock_eval, n_values)

    if got == want:
        out["optimal_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"
    return out
