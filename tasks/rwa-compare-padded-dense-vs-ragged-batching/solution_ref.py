import math


def _softmax(x):
    if isinstance(x[0], list):
        rows = len(x)
        cols = len(x[0])
        res = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            row_max = -float('inf')
            for j in range(cols):
                if x[i][j] > row_max:
                    row_max = x[i][j]
            row_sum = 0.0
            for j in range(cols):
                val = math.exp(x[i][j] - row_max)
                res[i][j] = val
                row_sum += val
            for j in range(cols):
                res[i][j] /= row_sum
        return res
    else:
        n = len(x)
        res = [0.0] * n
        row_max = -float('inf')
        for i in range(n):
            if x[i] > row_max:
                row_max = x[i]
        row_sum = 0.0
        for i in range(n):
            val = math.exp(x[i] - row_max)
            res[i] = val
            row_sum += val
        for i in range(n):
            res[i] /= row_sum
        return res


def ragged_attention_compare(sequences: list[list[list[float]]]) -> tuple[list[list[list[float]]], float]:
    outputs = []
    for x in sequences:
        n = len(x)
        d = len(x[0])

        scores = [[0.0] * n for _ in range(n)]
        sqrt_d = math.sqrt(d)
        for i in range(n):
            for j in range(n):
                val = 0.0
                for k in range(d):
                    val += x[i][k] * x[j][k]
                scores[i][j] = val / sqrt_d

        probs = _softmax(scores)

        out = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d):
                val = 0.0
                for k in range(n):
                    val += probs[i][k] * x[k][j]
                out[i][j] = val
        outputs.append(out)

    lengths = []
    for x in sequences:
        lengths.append(len(x))

    max_len = 0
    for l in lengths:
        if l > max_len:
            max_len = l

    sum_sq = 0
    for l in lengths:
        sum_sq += l * l

    ratio = (len(sequences) * max_len ** 2) / sum_sq
    return outputs, ratio
