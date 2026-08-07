import sqlite3


def range_gpu_time(db_path):
    """Calculate overlapping GPU duration for each NVTX range."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, start_ns, end_ns FROM nvtx_events")
    ranges = cur.fetchall()
    cur.execute("SELECT start_ns, end_ns FROM kernel_events")
    kernels = cur.fetchall()
    conn.close()

    result = {}
    for r_name, r_start, r_end in ranges:
        total_overlap = 0
        for k_start, k_end in kernels:
            o_start = max(r_start, k_start)
            o_end = min(r_end, k_end)
            if o_start < o_end:
                total_overlap += (o_end - o_start)
        result[r_name] = result.get(r_name, 0) + total_overlap
    return result


def top_kernels_summary(db_path):
    """Return top 3 kernels by total time and their collective percentage share."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, start_ns, end_ns FROM kernel_events")
    rows = cur.fetchall()
    conn.close()

    totals = {}
    grand_total = 0
    for name, start_ns, end_ns in rows:
        dur = end_ns - start_ns
        totals[name] = totals.get(name, 0) + dur
        grand_total += dur

    sorted_kernels = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_kernels[:3]
    top_3_sum = sum(dur for _, dur in top_3)
    share = (top_3_sum / grand_total * 100.0) if grand_total > 0 else 0.0
    return top_3, share
