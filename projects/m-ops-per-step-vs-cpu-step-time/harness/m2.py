import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from launchbound.bounds import cut_ops_until_busy_fraction, find_min_ops_for_busy_fraction
    except ImportError as e:
        return {"pruning_matches": 0.0, "_note": f"Import error: {e}"}

    matched = 0
    total = len(ref.PRUNING_SPECS)
    for spec in ref.PRUNING_SPECS:
        want_cut = ref.cut_ops_until_busy_fraction(**spec)
        want_min = ref.find_min_ops_for_busy_fraction(
            spec["target_busy_fraction"],
            spec["cpu_launch_overhead_us"],
            spec["gpu_time_per_op_us"]
        )
        try:
            got_cut = cut_ops_until_busy_fraction(**spec)
            got_min = find_min_ops_for_busy_fraction(
                spec["target_busy_fraction"],
                spec["cpu_launch_overhead_us"],
                spec["gpu_time_per_op_us"]
            )
        except Exception as e:
            return {"pruning_matches": 0.0, "_note": f"Execution error: {e}"}

        if got_cut == want_cut and got_min == want_min:
            matched += 1

    return {"pruning_matches": 1.0 if matched == total else 0.0}
