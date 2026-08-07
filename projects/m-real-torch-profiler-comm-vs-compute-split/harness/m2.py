import ref


def check(workdir):
    from profiler_analysis.overlap import compute_overlap_percentage

    total_err = 0.0
    count = 0
    for trace in ref.TEST_TRACES:
        want = ref.compute_overlap(trace)
        got = compute_overlap_percentage(trace)
        err = abs(got - want) / (max(abs(want), 1.0))
        total_err += err
        count += 1
    mean_err = total_err / max(1, count)
    return {"overlap_rel_err": float(mean_err)}
