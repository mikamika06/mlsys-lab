import math


def _gelu_scalar(val):
    return 0.5 * val * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val ** 3))))


def _gelu(x):
    def apply_recursive(sub_arr):
        if not isinstance(sub_arr, list):
            return _gelu_scalar(float(sub_arr))
        else:
            return [apply_recursive(item) for item in sub_arr]
    return apply_recursive(x)


def mlp_tensor_parallel(x: list[list[float]], w1_shards: list[list[list[float]]], b1_shards: list[list[float]], w2_shards: list[list[list[float]]], b2: list[float]) -> list[list[float]]:
    """Column-parallel then row-parallel tensor-parallel MLP forward pass."""
    m = len(x)
    d = len(x[0])
    d_out = len(b2)
    y_data = [[0.0 for _ in range(d_out)] for _ in range(m)]
    for w1_shard, b1_shard, w2_shard in zip(w1_shards, b1_shards, w2_shards):
        h = len(w1_shard[0])
        prod1 = [[0.0 for _ in range(h)] for _ in range(m)]
        for i in range(m):
            for j in range(h):
                s = 0.0
                for k in range(d):
                    s += x[i][k] * w1_shard[k][j]
                prod1[i][j] = s + b1_shard[j]
        a = _gelu(prod1)
        prod2 = [[0.0 for _ in range(d_out)] for _ in range(m)]
        for i in range(m):
            for j in range(d_out):
                s = 0.0
                for k in range(h):
                    s += a[i][k] * w2_shard[k][j]
                prod2[i][j] = s
        for i in range(m):
            for j in range(d_out):
                y_data[i][j] += prod2[i][j]
    for i in range(m):
        for j in range(d_out):
            y_data[i][j] += b2[j]
    return y_data
