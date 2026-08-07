import ref


def check(workdir):
    from irconv.precision import calculate_fp16_fp32_ratio, estimate_ir_size

    out = {"size_ratio_matched": 0.0, "bytes_calculated": 0.0}

    graph = ref.SAMPLE_GRAPHS[0]
    fp32_est = estimate_ir_size(graph, "FP32")
    fp16_est = estimate_ir_size(graph, "FP16")

    weights_count = 128 * 64 + 64
    expected_fp32_bin = weights_count * 4
    expected_fp16_bin = weights_count * 2

    if fp32_est["bin_bytes"] == expected_fp32_bin and fp16_est["bin_bytes"] == expected_fp16_bin:
        out["bytes_calculated"] = 1.0
    else:
        out["_note"] = f"Expected bin bytes FP32={expected_fp32_bin}, FP16={expected_fp16_bin}, got {fp32_est}, {fp16_est}"

    ratio = calculate_fp16_fp32_ratio(graph)
    expected_ratio = fp16_est["total_bytes"] / fp32_est["total_bytes"]

    if abs(ratio - expected_ratio) < 1e-4 and 0.45 < ratio < 0.55:
        out["size_ratio_matched"] = 1.0
    else:
        out["_note"] = f"Ratio mismatch: expected {expected_ratio:.4f}, got {ratio:.4f}"

    return out
