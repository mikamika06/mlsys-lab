def find_io_bottleneck(osrt_summary_rows):
    worst_op = None
    max_wait = -1.0
    for row in osrt_summary_rows:
        op_name = row["name"]
        total_time = row["total_time_ms"]
        if total_time > max_wait:
            max_wait = total_time
            worst_op = op_name
    return worst_op
