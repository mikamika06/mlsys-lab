def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        raise ValueError("median of nothing")
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def quartiles(xs):
    s = sorted(xs)
    n = len(s)
    if n < 2:
        return (s[0], s[0], s[0]) if n else (0.0, 0.0, 0.0)
    m = n // 2
    lower = s[:m]
    upper = s[m + 1:] if n % 2 else s[m:]
    return median(lower), median(s), median(upper)


def iqr(xs):
    q1, _, q3 = quartiles(xs)
    return q3 - q1


def separable(a, b):
    """1 when the two sample sets' inter-quartile ranges do not overlap."""
    if len(a) < 2 or len(b) < 2:
        return 0
    a1, _, a3 = quartiles(a)
    b1, _, b3 = quartiles(b)
    return 1 if (a3 < b1 or b3 < a1) else 0


def spread(xs):
    m = median(xs)
    return iqr(xs) / m if m else 0.0
