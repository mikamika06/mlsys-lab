import sys


def _oracle_kind(s):
    n = len(s)
    target = sys.getsizeof(s)
    candidates = {
        1: sys.getsizeof("é" * n),
        2: sys.getsizeof("Ā" * n),
        4: sys.getsizeof("😀" * n),
    }
    if target == sys.getsizeof("A" * n):
        return 1
    for kind, size in candidates.items():
        if target == size:
            return kind
    raise RuntimeError("unable to infer CPython unicode storage kind")


def grade(sol, fx) -> dict:
    cases = [
        ["hello", "world"],
        ["café", "naïve", "ÿ"],
        ["Ā", "Ω", "Ж"],
        ["😀", "🦊", "𐍈"],
        ["a", "é", "Ā", "😀", "abc123"],
        ["x" * 17, "é" * 17, "Ā" * 17, "😀" * 17],
    ]
    ok = 1.0
    for strings in cases:
        try:
            expected = [_oracle_kind(s) for s in strings]
            got = sol.classify_storage_kind(list(strings))
        except Exception:
            ok = 0.0
            break
        if list(got) != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
