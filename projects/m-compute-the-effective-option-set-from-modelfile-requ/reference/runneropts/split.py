def classify_options(traces):
    load_time = []
    sample_time = []
    for opt, data in traces.items():
        if data.get("load_duration_changed", False):
            load_time.append(opt)
        else:
            sample_time.append(opt)
    return sorted(load_time), sorted(sample_time)
