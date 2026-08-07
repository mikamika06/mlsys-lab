import math


def scaled_dot_product_attention(q, k, v):
    n = len(q)
    d = len(q[0])
    m = len(k)
    dv = len(v[0])

    scale = math.sqrt(d)
    logits = []
    for i in range(n):
        row = []
        for j in range(m):
            dot = 0.0
            for l in range(d):
                dot += q[i][l] * k[j][l]
            row.append(dot / scale)
        logits.append(row)

    weights = []
    for i in range(n):
        max_val = max(logits[i])
        exps = [math.exp(val - max_val) for val in logits[i]]
        sum_exps = sum(exps)
        weights.append([e / sum_exps for e in exps])

    output = []
    for i in range(n):
        row = []
        for j in range(dv):
            val = 0.0
            for l in range(m):
                val += weights[i][l] * v[l][j]
            row.append(val)
        output.append(row)

    return output
