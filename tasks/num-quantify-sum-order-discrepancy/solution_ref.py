def sum_order_discrepancy(arr):
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)

    n = a.shape[0]
    vals = []
    for i in range(n):
        vals.append(float(a[i]))

    sorted_vals = sorted(vals)

    s_asc = 0.0
    for x in sorted_vals:
        s_asc += x

    s_desc = 0.0
    for i in range(n - 1, -1, -1):
        s_desc += sorted_vals[i]

    s_pair = 0.0
    for i in range(n):
        s_pair += float(a[i])

    return (float(s_asc), float(s_desc), float(s_pair))
