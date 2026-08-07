def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    from sparse_eval.dispatch import get_dispatch_info, select_execution_path

    res = {
        "detects_unsupported_pattern": 0.0,
        "detects_misaligned_shape": 0.0,
        "detects_small_batch": 0.0,
        "selects_sparse_kernel": 0.0,
    }

    cases = ref.test_shapes()
    for case in cases:
        path = select_execution_path(case["shape"], case["is_24"])
        info = get_dispatch_info(case["shape"], case["is_24"])
        if path == case["expected_path"] and info["path"] == path:
            if case["expected_path"] == "dense_unsupported_pattern":
                res["detects_unsupported_pattern"] = 1.0
            elif case["expected_path"] == "dense_fallback_misaligned":
                res["detects_misaligned_shape"] = 1.0
            elif case["expected_path"] == "dense_fallback_small_batch":
                res["detects_small_batch"] = 1.0
            elif case["expected_path"] == "sparse_24_tensor_core":
                res["selects_sparse_kernel"] = 1.0

    return res
