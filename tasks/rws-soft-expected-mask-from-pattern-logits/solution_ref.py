import math

def soft_expected_mask(logits: list[list[float]], patterns: list[list[float]]) -> list[list[float]]:
    out = []
    B = len(logits)
    if B == 0:
        return out

    P = len(logits[0])
    D = len(patterns[0]) if len(patterns) > 0 else 0

    for i in range(B):
        row = logits[i]

        max_val = row[0]
        for j in range(1, P):
            if row[j] > max_val:
                max_val = row[j]

        exp_row = []
        sum_exp = 0.0
        for j in range(P):
            val = math.exp(row[j] - max_val)
            exp_row.append(val)
            sum_exp += val

        probs = []
        for j in range(P):
            probs.append(exp_row[j] / sum_exp)

        out_row = []
        for k in range(D):
            acc = 0.0
            for j in range(P):
                acc += probs[j] * patterns[j][k]
            out_row.append(acc)

        out.append(out_row)

    return out
