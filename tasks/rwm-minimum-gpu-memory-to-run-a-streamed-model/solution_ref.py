def min_gpu_memory(layer_bytes: list[int], K: int, activation_buffer: int) -> int:
    """
    Minimum GPU memory to run a layer-streamed model.
    """
    w = []
    for x in layer_bytes:
        w.append(int(x))

    n = len(w)
    k = K
    if k < 1:
        k = 1
    if k > n:
        k = n

    csum = [0] * (n + 1)
    acc = 0
    for i in range(n):
        acc += w[i]
        csum[i + 1] = acc

    peak = csum[k] - csum[0]
    for i in range(1, n - k + 1):
        val = csum[i + k] - csum[i]
        if val > peak:
            peak = val

    return peak + int(activation_buffer)
