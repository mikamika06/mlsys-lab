import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from edgelat.profiler import LatencyProfiler

    m = {"warmup_ok": 0.0}
    data = ref.get_mock_data()
    p = LatencyProfiler(data)
    res = p.filter_warmup_and_throttle(2, 2)
    expected = ref.np.array(data[2:-2])
    if ref.np.allclose(res, expected):
        m["warmup_ok"] = 1.0
    return m
