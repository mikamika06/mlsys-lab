import ref

def check(workdir):
    m = {"kernels_matched": 0.0}
    try:
        from profiler.correlation import correlate_kernels
        ranges = ref.get_sample_trace()
        kernels = ref.get_sample_kernels()
        res = correlate_kernels(ranges, kernels)
        if isinstance(res, list) and len(res) > 0:
            m["kernels_matched"] = 1.0
    except Exception:
        pass
    return m
