import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from edgelat.profiler import LatencyProfiler

    m = {"cold_start_ok": 0.0}
    data = ref.get_mock_data()
    p = LatencyProfiler(data)
    cold = p.measure_cold_start()
    if cold == 25.0:
        m["cold_start_ok"] = 1.0
    return m
