import math


def entropy_inflation_ratio(Q: list[list[float]], K: list[list[float]]) -> float:
    N = len(Q)
    d = len(Q[0])
    M = len(K)

    sqrt_d = math.sqrt(d)

    scores = []
    for i in range(N):
        row = []
        for j in range(M):
            s = 0.0
            for k in range(d):
                s += float(Q[i][k]) * float(K[j][k])
            row.append(s)
        scores.append(row)

    def calc_mean_entropy(scale: float) -> float:
        total_entropy = 0.0
        for i in range(N):
            row = scores[i]
            max_val = row[0] / scale
            for j in range(1, M):
                val = row[j] / scale
                if val > max_val:
                    max_val = val

            exp_vals = []
            exp_sum = 0.0
            for j in range(M):
                val = row[j] / scale
                e = math.exp(val - max_val)
                exp_vals.append(e)
                exp_sum += e

            row_entropy = 0.0
            for j in range(M):
                p = exp_vals[j] / exp_sum
                row_entropy += p * math.log(p + 1e-12)

            total_entropy += -row_entropy

        return total_entropy / N

    scaled_mean = calc_mean_entropy(sqrt_d)
    unscaled_mean = calc_mean_entropy(1.0)

    return float(scaled_mean / unscaled_mean)
