def _lcp(a: str, b: str) -> int:
    """Return length of longest common prefix of a and b."""
    i = 0
    la, lb = len(a), len(b)
    while i < la and i < lb and a[i] == b[i]:
        i += 1
    return i

def _oracle(query: str, cache: list[str]) -> str:
    max_lcp = max((_lcp(query, c) for c in cache), default=0)
    if max_lcp == len(query):
        return "full"
    if 0 < max_lcp < len(query):
        return "partial"
    return "miss"

def grade(sol, fx) -> dict:
    cases = [
        ("abc", ["abcd", "abx"]),
        ("abcde", ["abxyz", "a"]),
        ("hello", ["world", "test"]),
        ("", ["anything"]),
        ("foo", ["foobar", "fo"]),
        ("bar", ["baz", "bark"]),
        ("abc", ["abx", "a"]),
        ("abcd", ["ab", "a"]),
    ]
    ok = 1.0
    for query, cache in cases:
        try:
            got = sol.classify_query(query, cache)
        except Exception:
            return {"exact_match": 0.0}
        ref = _oracle(query, cache)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
