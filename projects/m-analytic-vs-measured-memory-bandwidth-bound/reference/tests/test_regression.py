from bwbound.analytic import compute_analytic_bound


def test_analytic_dtype_scaling():
    """Verify that analytic bounds scale correctly with mixed precision dtypes."""
    tensors_fp32 = [([1024, 4096], "fp32"), ([4096], "fp32")]
    tensors_int4 = [([1024, 4096], "int4"), ([4096], "fp16")]
    flops = 2 * 1024 * 4096
    peak_bw = 1000.0
    peak_tf = 100.0

    res_fp32 = compute_analytic_bound(tensors_fp32, flops, peak_bw, peak_tf)
    res_int4 = compute_analytic_bound(tensors_int4, flops, peak_bw, peak_tf)

    assert res_fp32["total_bytes"] > res_int4["total_bytes"]
    assert res_fp32["time_mem_sec"] > res_int4["time_mem_sec"]
