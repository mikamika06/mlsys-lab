def sum_order_discrepancy(arr: list[float]) -> tuple[float, float, float]:
    vals = [float(x) for x in arr]
    n = len(vals)

    sorted_vals = sorted(vals)

    s_asc = 0.0
    for x in sorted_vals:
        s_asc += x

    s_desc = 0.0
    for i in range(n - 1, -1, -1):
        s_desc += sorted_vals[i]

    s_pair = 0.0
    for i in range(n):
        s_pair += vals[i]

    return (float(s_asc), float(s_desc), float(s_pair))
