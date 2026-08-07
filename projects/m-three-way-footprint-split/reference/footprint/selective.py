def selective_registration_win(all_kernels, used_ops):
    """Calculate binary size and memory savings from selective op registration."""
    used_set = set(used_ops)
    total_binary_before = 0
    total_binary_after = 0
    total_table_before = 0
    total_table_after = 0

    pruned_ops = []

    for k in all_kernels:
        op_name = k["op"]
        code_size = k.get("code_bytes", 0)
        table_size = k.get("table_bytes", 0)

        total_binary_before += code_size
        total_table_before += table_size

        if op_name in used_set:
            total_binary_after += code_size
            total_table_after += table_size
        else:
            pruned_ops.append(op_name)

    binary_saved = total_binary_before - total_binary_after
    table_saved = total_table_before - total_table_after
    total_saved = binary_saved + table_saved

    pct_binary_saved = (binary_saved / total_binary_before * 100.0) if total_binary_before > 0 else 0.0

    return {
        "binary_bytes_before": total_binary_before,
        "binary_bytes_after": total_binary_after,
        "binary_bytes_saved": binary_saved,
        "table_bytes_saved": table_saved,
        "total_bytes_saved": total_saved,
        "percent_binary_saved": pct_binary_saved,
        "pruned_ops": sorted(pruned_ops)
    }
