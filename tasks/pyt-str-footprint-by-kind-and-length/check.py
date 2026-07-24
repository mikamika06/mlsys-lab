import sys


def _oracle(lengths, max_codepoints):
    out = []
    for n, cp in zip(lengths, max_codepoints):
        if n == 0:
            s = ""
        else:
            ch = chr(cp)
            s = ch * n
        out.append(sys.getsizeof(s))
    return out


def grade(sol, fx) -> dict:
    cases = [
        ([0, 1, 8], [0, 65, 127]),
        ([1, 2, 16], [128, 233, 255]),
        ([1, 3, 20], [256, 1024, 65535]),
        ([1, 4, 12], [65536, 128512, 1114111]),
        ([5, 50, 500], [65, 233, 128512]),
    ]

    ok = 1.0
    for lengths, cps in cases:
        try:
            got = list(sol.str_footprint(list(lengths), list(cps)))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(lengths, cps)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
