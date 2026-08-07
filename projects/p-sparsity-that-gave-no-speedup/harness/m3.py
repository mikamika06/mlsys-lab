def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    from sparse_eval.benchmark import simulate_kernel_metrics

    res = {
        "sparse_flops_halved": 0.0,
        "dense_flops_correct": 0.0,
        "memory_bytes_accounted": 0.0,
    }

    shape = (64, 128, 256)
    m_sparse = simulate_kernel_metrics(shape, "sparse_24_tensor_core")
    m_dense = simulate_kernel_metrics(shape, "dense_unsupported_pattern")

    expected_dense_flops = 2 * 64 * 128 * 256
    expected_sparse_flops = 64 * 128 * 256

    if m_dense["flops"] == expected_dense_flops:
        res["dense_flops_correct"] = 1.0

    if m_sparse["flops"] == expected_sparse_flops and m_sparse["flops"] == m_dense["flops"] // 2:
        res["sparse_flops_halved"] = 1.0

    exp_sparse_bytes = int(0.5 * 64 * 256 * 2 + 64 * 64) + (64 * 256 + 64 * 128 + 128 * 256) * 2
    if m_sparse["total_bytes"] == exp_sparse_bytes:
        res["memory_bytes_accounted"] = 1.0

    return res
