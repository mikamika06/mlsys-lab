def online_moments(values):
    n = 0
    mean = 0.0
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0

    for x in values:
        x = float(x)
        n1 = n
        n += 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1

        mean += delta_n

        m4 += (
            term1 * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * m2
            - 4 * delta_n * m3
        )
        m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
        m2 += term1

    if n == 0 or m2 == 0.0:
        return 0.0, 0.0

    var = m2 / n
    skew = (m3 / n) / (var ** 1.5)
    kurt = (m4 / n) / (var ** 2) - 3.0
    return float(skew), float(kurt)
