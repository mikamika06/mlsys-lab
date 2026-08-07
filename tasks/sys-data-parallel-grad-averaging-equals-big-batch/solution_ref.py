def data_parallel_grad_avg(X: list[list[float]], y: list[float], num_shards: int) -> list[float]:
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    if n % num_shards != 0:
        raise ValueError("n must be divisible by num_shards")
    shard_size = n // num_shards

    grads = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size
        Xi = X[start:end]
        yi = y[start:end]

        X_T_y = [0.0] * d
        for j in range(d):
            s_val = 0.0
            for k in range(shard_size):
                s_val += Xi[k][j] * yi[k]
            X_T_y[j] = s_val

        grad_i = [(2.0 / shard_size) * val for val in X_T_y]
        grads.append(grad_i)

    global_grad = [0.0] * d
    for j in range(d):
        s_sum = 0.0
        for i in range(num_shards):
            s_sum += grads[i][j]
        global_grad[j] = s_sum / num_shards

    return global_grad
