def compute_nvtx_gpu_time(nvtx_rows, kernel_rows):
    results = []
    for r in nvtx_rows:
        r_start = r["start"]
        r_end = r["end"]
        r_name = r["name"]
        total_time = 0.0
        for k in kernel_rows:
            k_start = k["start"]
            k_end = k["end"]
            overlap_start = max(r_start, k_start)
            overlap_end = min(r_end, k_end)
            if overlap_start < overlap_end:
                total_time += (overlap_end - overlap_start)
        results.append({"name": r_name, "gpu_time": total_time})
    return results


def top_kernels(kernel_rows, n=3):
    totals = {}
    for k in kernel_rows:
        name = k["name"]
        dur = k["end"] - k["start"]
        totals[name] = totals.get(name, 0.0) + dur
    grand_total = sum(totals.values())
    sorted_kernels = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    top = sorted_kernels[:n]
    out = []
    for name, t in top:
        share = (t / grand_total) if grand_total > 0 else 0.0
        out.append({"name": name, "total_time": t, "share": share})
    return out
