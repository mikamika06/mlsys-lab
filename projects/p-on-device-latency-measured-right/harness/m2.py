import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from edgelat.profiler import LatencyProfiler

    m = {"separation_ok": 0.0}
    data = ref.get_mock_data()
    p = LatencyProfiler(data)
    first, steady = p.separate_first_and_steady()
    if first == 25.0 and isinstance(steady, float):
        m["separation_ok"] = 1.0
    return m
