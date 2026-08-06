def sq_dist_expansion(a: list[float], b: list[float]) -> float:
    dot_aa = 0.0
    for i in range(len(a)):
        dot_aa += float(a[i]) * float(a[i])

    dot_bb = 0.0
    for i in range(len(b)):
        dot_bb += float(b[i]) * float(b[i])

    dot_ab = 0.0
    for i in range(len(a)):
        dot_ab += float(a[i]) * float(b[i])

    return float(dot_aa + dot_bb - 2.0 * dot_ab)
