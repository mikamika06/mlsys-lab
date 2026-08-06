import math


def z_loss(logits: list[list[float]], targets: list[int], lambda_: float) -> list[float]:
    N = len(logits)
    C = len(logits[0])
    out = []
    for i in range(N):
        m = logits[i][0]
        for j in range(1, C):
            val = logits[i][j]
            if val > m:
                m = val

        s = 0.0
        for j in range(C):
            s += math.exp(logits[i][j] - m)

        lse = m + math.log(s)
        ce = -logits[i][targets[i]] + lse
        out.append(ce + lambda_ * (lse ** 2))

    return out
