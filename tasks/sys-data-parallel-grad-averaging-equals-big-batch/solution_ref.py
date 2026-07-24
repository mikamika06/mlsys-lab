import numpy as np

def data_parallel_grad_avg(X, y, num_shards):
    n, d = X.shape
    if n % num_shards != 0:
        raise ValueError("n must be divisible by num_shards")
    shard_size = n // num_shards
    grads = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size
        Xi = X[start:end]
        yi = y[start:end]
        grad_i = 2.0 / shard_size * Xi.T @ yi
        grads.append(grad_i)
    return sum(grads) / num_shards
