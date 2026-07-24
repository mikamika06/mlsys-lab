def model_kernel(n: int, k: int):
    latency = 4
    rolled = n * latency
    unrolled = ((n + k - 1) // k) * latency
    addresses = [index * 8 for index in range(n)]
    return rolled, unrolled, addresses
