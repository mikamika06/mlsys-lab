def check(workdir):
    from moe.profiler import measure_activation_distribution
    import ref

    m = {"distribution_ok": 0.0}
    traces = ref.get_sample_traces()
    res = measure_activation_distribution(traces)
    ref_res = {1: 2/7, 2: 2/7, 3: 2/7, 4: 1/7}

    if set(res.keys()) == set(ref_res.keys()):
        match = True
        for k in ref_res:
            if abs(res[k] - ref_res[k]) > 1e-5:
                match = False
        if match:
            m["distribution_ok"] = 1.0
    return m
