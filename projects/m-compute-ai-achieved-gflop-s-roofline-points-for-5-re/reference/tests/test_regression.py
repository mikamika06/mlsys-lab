from roofline.calc import compute_roofline_points, classify_kernels, compare_attention, find_limiter


def test_compute_roofline_points():
    kernels = [{"name": "gemm", "flops": 2e9, "bytes": 1e7, "time_s": 0.001}]
    res = compute_roofline_points(kernels, peak_flops=1000.0, peak_bw=100.0)
    assert len(res) == 1
    assert res[0]["bound"] == "compute"


def test_classify_kernels():
    metrics = [{"bound": "compute"}, {"bound": "memory"}]
    assert classify_kernels(metrics) == ["compute", "memory"]


def test_compare_attention():
    std = {"ai": 1.0}
    flash = {"ai": 10.0}
    assert compare_attention(std, flash) == "flash"


def test_find_limiter():
    metric = {"ai": 0.1, "gflops": 1.0}
    limiter = find_limiter(metric, peak_flops=1000.0, peak_bw=100.0)
    assert limiter == "latency_or_overhead_bound"
