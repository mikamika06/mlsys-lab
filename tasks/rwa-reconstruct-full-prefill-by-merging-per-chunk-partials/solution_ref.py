import math


def merge_chunk_partials(ms, ls, os):
    n_chunks = len(ms)

    M = ms[0]
    for i in range(1, n_chunks):
        if ms[i] > M:
            M = ms[i]

    scale = [0.0] * n_chunks
    for i in range(n_chunks):
        scale[i] = math.exp(ms[i] - M)

    L = 0.0
    for i in range(n_chunks):
        L += ls[i] * scale[i]

    d = len(os[0])
    O = [0.0] * d
    for j in range(d):
        acc = 0.0
        for i in range(n_chunks):
            acc += os[i][j] * scale[i]
        O[j] = acc / L

    return O
