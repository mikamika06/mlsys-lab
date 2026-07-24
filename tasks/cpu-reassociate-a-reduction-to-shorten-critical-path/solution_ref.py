def reassociated_sum_trace(values, base_addr):
    addrs = [base_addr + i * 8 for i in range(len(values))]

    partials = [float(x) for x in values]
    depth = 0
    while len(partials) > 1:
        nxt = []
        i = 0
        while i < len(partials):
            if i + 1 < len(partials):
                nxt.append(partials[i] + partials[i + 1])
            else:
                nxt.append(partials[i])
            i += 2
        partials = nxt
        depth += 1

    total = partials[0] if partials else 0.0

    return {
        "total": float(total),
        "addrs": addrs,
        "critical_path": depth,
    }
