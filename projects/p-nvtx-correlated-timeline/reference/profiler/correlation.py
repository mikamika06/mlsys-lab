def correlate_kernels(ranges, kernels):
    matched = []
    for r in ranges:
        r_kernels = [k for k in kernels if r["start"] <= k["start"] and k["end"] <= r["end"]]
        matched.append({"phase": r["name"], "kernels": r_kernels})
    return matched
