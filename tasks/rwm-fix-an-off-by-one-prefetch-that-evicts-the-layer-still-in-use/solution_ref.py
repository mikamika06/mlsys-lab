import math


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for l in range(k):
            Ail = Ai[l]
            Bl = B[l]
            for j in range(n):
                Ci[j] += Ail * Bl[j]
    return C


def _transpose(A: list[list[float]]) -> list[list[float]]:
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _attention(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> list[list[float]]:
    d = len(q[0])
    kt = _transpose(k)
    qk = _matmul(q, kt)
    scale = math.sqrt(d)

    scores = []
    for row in qk:
        scaled_row = [x / scale for x in row]
        max_val = max(scaled_row)
        exp_row = [math.exp(x - max_val) for x in scaled_row]
        sum_exp = sum(exp_row)
        softmax_row = [x / sum_exp for x in exp_row]
        scores.append(softmax_row)

    return _matmul(scores, v)


def scheduled_attention(
    layers: list[tuple[list[list[float]], list[list[float]]]],
    Qs: list[list[list[float]]],
    Ks,
    Vs,
) -> list[list[list[float]]]:
    outputs = []
    cache = [None, None]

    for i, q in enumerate(Qs):
        slot = i % 2
        cache[slot] = layers[i]
        k, v = cache[slot]
        outputs.append(_attention(q, k, v))

    return outputs
