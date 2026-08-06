import numpy as np

def pairwise_sum(arr):
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0

    def rec(start, end):
        length = end - start
        if length <= 1024:
            total = 0.0
            for i in range(start, end):
                total += float(arr[i])
            return total
        mid = (start + end) // 2
        left = rec(start, mid)
        right = rec(mid, end)
        return left + right

    return rec(0, n)
