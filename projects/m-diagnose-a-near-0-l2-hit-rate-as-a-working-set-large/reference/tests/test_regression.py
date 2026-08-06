from profiler.tiling import verify_tiling_dram_reduction


def test_tiling_dram_reduction_logic():
    res = verify_tiling_dram_reduction(1000000, 200000, 2.0)
    assert res["dram_bytes_reduced"] is True
    assert res["speedup_explained_by_dram"] is True


def test_tiling_no_reduction():
    res = verify_tiling_dram_reduction(200000, 1000000, 2.0)
    assert res["dram_bytes_reduced"] is False
    assert res["speedup_explained_by_dram"] is False
