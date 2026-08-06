from kernelperf.gflops import rank_kernels_by_gflops


def test_ranking_order():
    kernels = [
        {"name": "a", "compute_pct": 10.0},
        {"name": "b", "compute_pct": 90.0}
    ]
    ranked = rank_kernels_by_gflops(kernels, 312.0)
    assert ranked == ["b", "a"]
