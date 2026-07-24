import itertools
import re


def _oracle(regex, vocab, max_len):
    accepted = set()
    for length in range(max_len + 1):
        for parts in itertools.product(vocab, repeat=length):
            s = "".join(parts)
            if re.fullmatch(regex, s):
                accepted.add(s)
    return sorted(accepted)


def grade(sol, fx) -> dict:
    cases = [
        (r"ab+", ["a", "b"], 3),
        (r"(cat|dog)", ["c", "a", "t", "d", "o", "g"], 3),
        (r"a[bc]d", ["a", "b", "c", "d"], 4),
        (r"0[01]*1", ["0", "1"], 4),
        (r"x*y", ["x", "y"], 3),
    ]

    ok = 1.0
    for regex, vocab, max_len in cases:
        expected = _oracle(regex, vocab, max_len)
        try:
            got = sol.enumerate_accepted_strings(regex, list(vocab), max_len)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
