from fractions import Fraction
from math import gcd


def _oracle(prefill_tps, decode_tps, input_len, output_len):
    ratio = Fraction(decode_tps * input_len, prefill_tps * output_len)
    p = ratio.numerator
    d = ratio.denominator
    g = gcd(p, d)
    return (p // g, d // g)


def grade(sol, fx) -> dict:
    cases = [
        (4000, 200, 1000, 500),
        (1200, 300, 2048, 256),
        (500, 750, 100, 900),
        (16384, 512, 512, 128),
        (1000, 1000, 1, 1),
    ]
    ok = 1.0
    for case in cases:
        expected = _oracle(*case)
        try:
            got = sol.balanced_pd_ratio(*case)
            got = (int(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
