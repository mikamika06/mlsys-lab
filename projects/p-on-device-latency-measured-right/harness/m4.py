import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from edgelat.profiler import LatencyProfiler

    m = {"sample_size_ok": 0.0}
    data = ref.get_mock_data()
    p = LatencyProfiler(data)
    n = p.required_sample_size(0.05, 1.0)
    if isinstance(n, int) and n > 0:
        m["sample_size_ok"] = 1.0
    return m
