def _ref_edit_distance(source: str, target: str) -> int:
    """Canonical Wagner–Fischer DP — the oracle."""
    m, n = len(source), len(target)
    # Two-row DP to keep memory O(min(m,n))
    if m < n:
        source, target = target, source
        m, n = n, m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        si = source[i - 1]
        for j in range(1, n + 1):
            if si == target[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]

def grade(sol, fx) -> dict:
    cases = [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
        ("kitten", "sitting"),
        ("saturday", "sunday"),
        ("abcdef", "azcedf"),
        ("intention", "execution"),
        ("algorithm", "altruistic"),
        ("hello", "hullo"),
        ("same", "same"),
        ("a", "b"),
        ("ab", "ba"),
        ("abcde", "edcba"),
        ("sunday", "saturday"),
        ("sitting", "kitten"),
        ("x" * 50, "y" * 50),
        ("a" * 200, "a" * 199 + "b"),
        ("abc" * 30, "bac" * 30),
        ("", "x" * 100),
        ("x" * 100, ""),
    ]
    ok = 1.0
    for source, target in cases:
        try:
            got = sol.edit_distance(source, target)
        except Exception:
            ok = 0.0
            break
        expected = _ref_edit_distance(source, target)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
