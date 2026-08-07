import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from edgelat.profiler import LatencyProfiler

    m = {"sessions_overlap": 0.0}
    sessions = ref.get_multi_session_data()
    p = LatencyProfiler(sessions[0])
    intervals = p.multi_session_intervals(sessions)
    if len(intervals) == 3 and all(isinstance(i, tuple) and len(i) == 2 for i in intervals):
        m["sessions_overlap"] = 1.0
    return m
