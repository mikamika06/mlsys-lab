def sum_order_discrepancy(arr):
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)

    sorted_asc = np.sort(a)
    sorted_desc = np.flip(sorted_asc)

    s_asc = np.sum(sorted_asc)
    s_desc = np.sum(sorted_desc)
    s_pair = np.add.reduce(a)

    return (float(s_asc), float(s_desc), float(s_pair))
