import numpy as np

def reduce_scatter(data, op="sum"):
    n = len(data)
    k = len(data[0]) // n
    result = []
    for j in range(n):
        res_chunk = np.empty(k, dtype=data[0].dtype)
        for c in range(k):
            if op == "sum":
                acc = 0.0
                for i in range(n):
                    acc += data[i][j * k + c]
                res_chunk[c] = acc
            else:
                m = data[0][j * k + c]
                for i in range(1, n):
                    val = data[i][j * k + c]
                    if val > m:
                        m = val
                res_chunk[c] = m
        result.append(res_chunk)
    return result

def all_gather(data):
    combined = np.concatenate(data)
    return [combined.copy() for _ in range(len(data))]
