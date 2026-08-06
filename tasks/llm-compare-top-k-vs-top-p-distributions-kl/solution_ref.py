import math


def top_k_top_p_kl(
    logits: list[list[float]], k: int, p: float
) -> float:
    rows = len(logits)
    cols = len(logits[0])

    eps = 1e-12
    total_kl = 0.0

    for r in range(rows):
        row = logits[r]

        max_val = float(row[0])
        for c in range(1, cols):
            val = float(row[c])
            if val > max_val:
                max_val = val

        row_exps = []
        row_sum = 0.0
        for c in range(cols):
            val = math.exp(float(row[c]) - max_val)
            row_exps.append(val)
            row_sum += val

        row_probs = []
        for c in range(cols):
            row_probs.append(row_exps[c] / row_sum)

        order = sorted(range(cols), key=lambda c: row_probs[c])[::-1]

        row_topk = [0.0] * cols
        for idx in order[:k]:
            row_topk[idx] = row_probs[idx]

        row_topp = [0.0] * cols
        accum = 0.0
        for idx in order:
            row_topp[idx] = row_probs[idx]
            accum += row_probs[idx]
            if accum >= p:
                break

        topk_sum = 0.0
        for c in range(cols):
            topk_sum += row_topk[c]

        for c in range(cols):
            row_topk[c] = row_topk[c] / topk_sum

        topp_sum = 0.0
        for c in range(cols):
            topp_sum += row_topp[c]

        for c in range(cols):
            row_topp[c] = row_topp[c] / topp_sum

        row_kl = 0.0
        for c in range(cols):
            row_kl += row_topk[c] * (
                math.log(row_topk[c] + eps) - math.log(row_topp[c] + eps)
            )

        total_kl += row_kl

    return float(total_kl / rows)
