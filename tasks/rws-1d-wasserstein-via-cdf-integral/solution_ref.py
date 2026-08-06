import numpy as np


def wasserstein1_cdf_integral(u: np.ndarray, v: np.ndarray) -> float:
    """Exact 1-D Wasserstein-1 distance between two empirical samples of
    possibly unequal length, via the merged-CDF step integral
    W1(u,v) = sum_k |F_U(z_k) - F_V(z_k)| * (z_{k+1} - z_k).
    """

    def _flatten(arr):
        arr_np = np.asarray(arr, dtype=np.float64)
        res = []

        def _rec(a):
            if a.ndim == 0:
                res.append(float(a.item()))
            elif a.ndim == 1:
                for i in range(a.shape[0]):
                    res.append(float(a[i]))
            else:
                for i in range(a.shape[0]):
                    _rec(a[i])

        _rec(arr_np)
        return res

    def _merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = _merge_sort(arr[:mid])
        right = _merge_sort(arr[mid:])
        res = []
        i = 0
        j = 0
        len_l = len(left)
        len_r = len(right)
        while i < len_l and j < len_r:
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        while i < len_l:
            res.append(left[i])
            i += 1
        while j < len_r:
            res.append(right[j])
            j += 1
        return res

    u_list = _merge_sort(_flatten(u))
    v_list = _merge_sort(_flatten(v))

    n = len(u_list)
    m = len(v_list)

    all_values = []
    i = 0
    j = 0
    while i < n and j < m:
        if u_list[i] <= v_list[j]:
            all_values.append(u_list[i])
            i += 1
        else:
            all_values.append(v_list[j])
            j += 1
    while i < n:
        all_values.append(u_list[i])
        i += 1
    while j < m:
        all_values.append(v_list[j])
        j += 1

    total_sum = 0.0
    u_idx = 0
    v_idx = 0

    num_all = len(all_values)
    for k in range(num_all - 1):
        val = all_values[k]
        val_next = all_values[k + 1]
        delta = val_next - val

        while u_idx < n and u_list[u_idx] <= val:
            u_idx += 1
        while v_idx < m and v_list[v_idx] <= val:
            v_idx += 1

        u_cdf = u_idx / n
        v_cdf = v_idx / m

        diff = u_cdf - v_cdf
        if diff < 0.0:
            diff = -diff

        total_sum += diff * delta

    return float(total_sum)
