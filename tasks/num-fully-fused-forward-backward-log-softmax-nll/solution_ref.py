import math


def fused_log_softmax_nll(logits: list[list[float]], targets: list[int]):
    """Fused stable log-softmax forward + backward: mean NLL loss and its gradient."""
    n = len(logits)
    c = len(logits[0])

    m = [0.0] * n
    for i in range(n):
        mx = logits[i][0]
        for j in range(1, c):
            if logits[i][j] > mx:
                mx = logits[i][j]
        m[i] = mx

    shifted = [[0.0] * c for _ in range(n)]
    for i in range(n):
        for j in range(c):
            shifted[i][j] = logits[i][j] - m[i]

    lse = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(c):
            s += math.exp(shifted[i][j])
        lse[i] = m[i] + math.log(s)

    log_probs = [[0.0] * c for _ in range(n)]
    for i in range(n):
        for j in range(c):
            log_probs[i][j] = logits[i][j] - lse[i]

    sum_loss = 0.0
    for i in range(n):
        sum_loss += log_probs[i][targets[i]]
    loss = -float(sum_loss / n)

    probs = [[0.0] * c for _ in range(n)]
    for i in range(n):
        for j in range(c):
            probs[i][j] = math.exp(log_probs[i][j])

    dlogits = [[0.0] * c for _ in range(n)]
    for i in range(n):
        for j in range(c):
            dlogits[i][j] = probs[i][j]
    for i in range(n):
        dlogits[i][targets[i]] -= 1.0
    for i in range(n):
        for j in range(c):
            dlogits[i][j] /= n

    return loss, dlogits
