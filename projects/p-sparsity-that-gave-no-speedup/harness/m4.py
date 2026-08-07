def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    from sparse_eval.roofline import calculate_arithmetic_intensity, compute_roofline_bound, find_breakeven_m

    res = {
        "intensity_ratio_correct": 0.0,
        "roofline_bottleneck_identified": 0.0,
        "breakeven_m_computed": 0.0,
    }

    hw = ref.hw_config()
    dense_int = calculate_arithmetic_intensity(64, 4096, 4096, is_sparse=False)
    sparse_int = calculate_arithmetic_intensity(64, 4096, 4096, is_sparse=True)

    if sparse_int > 0 and dense_int > 0:
        res["intensity_ratio_correct"] = 1.0

    bound_small = compute_roofline_bound(2, 4096, 4096, hw["peak_tflops"], hw["bandwidth_gbps"], is_sparse=True)
    bound_large = compute_roofline_bound(256, 4096, 4096, hw["peak_tflops"], hw["bandwidth_gbps"], is_sparse=True)

    if bound_small["bottleneck"] == "memory" and bound_large["bottleneck"] == "compute":
        res["roofline_bottleneck_identified"] = 1.0

    m_break = find_breakeven_m(4096, 4096, hw["peak_tflops"], hw["bandwidth_gbps"])
    if 1 <= m_break <= 256:
        res["breakeven_m_computed"] = 1.0

    return res
