from actmem.scaling import measure_doubling_growth

def test_scaling_growth_strictly_increasing():
    vals = measure_doubling_growth(256, 1024, 4, 1)
    assert len(vals) == 4
    for i in range(1, len(vals)):
        assert vals[i] > vals[i-1]
