def check(workdir):
    import ref
    m = {"detected_64": 0.0, "detected_8": 0.0}
    try:
        from system import analysis
        spike_64 = analysis.find_first_spike(ref.get_recorded_losses(64))
        if spike_64 == 45:
            m["detected_64"] = 1.0
        spike_8 = analysis.find_first_spike(ref.get_recorded_losses(8))
        if spike_8 == -1:
            m["detected_8"] = 1.0
    except Exception:
        pass
    return m
