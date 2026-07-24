def merge_welford(a, b):
    na, ma, m2a = a
    nb, mb, m2b = b

    if na == 0:
        return nb, mb, m2b
    if nb == 0:
        return na, ma, m2a

    n = na + nb
    delta = mb - ma
    mean = ma + delta * (nb / n)
    M2 = m2a + m2b + delta * delta * (na * nb / n)
    return n, mean, M2
