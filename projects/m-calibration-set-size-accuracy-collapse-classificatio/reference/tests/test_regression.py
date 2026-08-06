from quantcollapse.analysis import reconstruct_scales, classify_collapse


def test_reconstruct_scales_basic():
    nodes = [{"min_val": -1.0, "max_val": 1.0, "levels": 256}]
    sc = reconstruct_scales(nodes)
    assert len(sc) == 1
    assert abs(sc[0] - 2.0 / 255.0) < 1e-5


def test_classify_collapse_basic():
    sizes = [10, 1000]
    variances = [50.0, 0.1]
    res = classify_collapse(sizes, variances, 1.0)
    assert res == ["collapse", "stable"]
