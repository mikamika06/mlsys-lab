def parse_ncu_diff(before: dict, after: dict) -> dict:
    b_read = before.get("dram__bytes_read.sum", 0)
    b_write = before.get("dram__bytes_write.sum", 0)
    a_read = after.get("dram__bytes_read.sum", 0)
    a_write = after.get("dram__bytes_write.sum", 0)

    b_total_dram = b_read + b_write
    a_total_dram = a_read + a_write

    dram_read_diff = b_read - a_read
    dram_write_diff = b_write - a_write
    total_dram_wasted_bytes = dram_read_diff + dram_write_diff

    b_inst = before.get("sm__inst_executed.sum", 0)
    a_inst = after.get("sm__inst_executed.sum", 0)
    inst_diff = b_inst - a_inst

    b_time = before.get("gpu__time_duration.sum", 0)
    a_time = after.get("gpu__time_duration.sum", 0)
    time_diff_ns = b_time - a_time

    dram_waste_ratio = total_dram_wasted_bytes / b_total_dram if b_total_dram > 0 else 0.0
    inst_waste_ratio = inst_diff / b_inst if b_inst > 0 else 0.0

    if dram_waste_ratio >= 0.15 and inst_waste_ratio >= 0.15:
        category = "dram_and_compute_waste"
    elif dram_waste_ratio >= 0.15:
        category = "dram_overfetch"
    elif inst_waste_ratio >= 0.15:
        category = "phantom_instructions"
    else:
        category = "minor_overhead"

    return {
        "before_dram_bytes": b_total_dram,
        "after_dram_bytes": a_total_dram,
        "before_time_ns": b_time,
        "after_time_ns": a_time,
        "dram_read_diff": dram_read_diff,
        "dram_write_diff": dram_write_diff,
        "total_dram_wasted_bytes": total_dram_wasted_bytes,
        "inst_diff": inst_diff,
        "time_diff_ns": time_diff_ns,
        "dram_waste_ratio": dram_waste_ratio,
        "inst_waste_ratio": inst_waste_ratio,
        "category": category,
    }
