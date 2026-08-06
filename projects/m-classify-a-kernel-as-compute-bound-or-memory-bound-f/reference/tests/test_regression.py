from roofline.classify import classify_kernel


def test_classify_boundary():
    res, intensity, crossover = classify_kernel(100, 10, 10.0, 1.0)
    assert res == "compute-bound"
    res2, _, _ = classify_kernel(1, 10, 10.0, 1.0)
    assert res2 == "memory-bound"
