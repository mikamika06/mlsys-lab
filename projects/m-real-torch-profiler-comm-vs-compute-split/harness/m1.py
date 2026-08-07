import ref


def check(workdir):
    from profiler_analysis.parser import parse_comm_compute_split

    matched = 0
    for i, trace in enumerate(ref.TEST_TRACES):
        want = ref.parse_split(trace)
        got = parse_comm_compute_split(trace)
        if isinstance(got, dict) and abs(got.get("compute_time", -1) - want["compute_time"]) < 1e-5 and abs(got.get("comm_time", -1) - want["comm_time"]) < 1e-5:
            matched += 1
    return {"split_match": float(matched)}
