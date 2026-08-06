def flops_neutral_batch_size(ar_batch_size: int, c: float, n: int) -> int:
    return int(ar_batch_size / (c * n + 1))
