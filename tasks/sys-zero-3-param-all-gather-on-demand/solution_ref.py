def zero3_linear_backward(weight_shards, x, grad_y):
    full_w = [row[:] for shard in weight_shards for row in shard]

    # Forward check dummy computation: x @ full_w.T
    n = len(x)
    d = len(x[0])
    m = len(full_w)

    # full_grad = grad_y.T @ x
    # grad_y has shape (n, m), grad_y.T has shape (m, n)
    # x has shape (n, d)
    # result full_grad has shape (m, d)
    full_grad = [[0.0.copy() if hasattr(0.0, 'copy') else 0.0 for _ in range(d)] for _ in range(m)]
    for i in range(m):
        for j in range(d):
            val = 0.0
            for k in range(n):
                val += grad_y[k][i] * x[k][j]
            full_grad[i][j] = val

    grads = []
    start = 0
    for shard in weight_shards:
        rows = len(shard)
        grads.append([full_grad[start + r][:] for r in range(rows)])
        start += rows
    return grads
