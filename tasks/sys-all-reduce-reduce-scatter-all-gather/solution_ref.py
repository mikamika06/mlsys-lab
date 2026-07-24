import numpy as np

def reduce_scatter(data, op="sum"):
    n = len(data)
    k = len(data[0]) // n
    _op = np.sum if op == "sum" else np.max
    result = []
    for j in range(n):
        chunks = [data[i][j * k : (j + 1) * k] for i in range(n)]
        stacked = np.stack(chunks, axis=0)       # (n, k)
        result.append(_op(stacked, axis=0))       # (k,)
    return result

def all_gather(data):
    combined = np.concatenate(data)               # (n*k,)
    return [combined.copy() for _ in range(len(data))]
