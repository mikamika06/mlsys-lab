import numpy as np

def _reference_classify(inv_freq):
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    n = len(inv_freq)
    if n < 2:
        return "None"
    idx = np.arange(n)
    std = 1 / (10000 ** (2 * idx / n))
    # PI: constant scaling factor
    k_vals = inv_freq / std
    k_mean = np.mean(k_vals)
    if abs(k_mean - 1.0) > 1e-6:
        return "PI"
    # NTK: geometric ratio sqrt of standard
    ratios = inv_freq[1:] / inv_freq[:-1]
    r_std_expected = std[1] / std[0]
    r_ntk_expected = np.sqrt(r_std_expected)
    if np.allclose(ratios, r_ntk_expected, atol=1e-6 * np.abs(r_ntk_expected)):
        return "NTK"
    # YaRN: linear progression in value
    diffs = np.diff(inv_freq)
    if np.std(diffs) < 1e-6 * np.mean(np.abs(diffs)):
        return "YaRN"
    return "None"

def grade(sol, fx) -> dict:
    cases = [
        ("standard", lambda n: 1 / (10000 ** (2 * np.arange(n) / n))),
        ("pi", lambda n: 3.0 * (1 / (10000 ** (2 * np.arange(n) / n)))),
        ("ntk", lambda n: 1 / (10000 ** (np.arange(n) / n))),
        ("yarn", lambda n: np.linspace(
            1 / (10000 ** (2 * 0 / n)),
            1 / (10000 ** (2 * (n-1) / n)),
            n
        )),
    ]
    ok = 1.0
    for name, fn in cases:
        inv = fn(8)
        try:
            got = sol.classify_extension(inv)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_classify(inv)
        if got != ref:
            ok = 0.0
    return {"exact_match": ok}
