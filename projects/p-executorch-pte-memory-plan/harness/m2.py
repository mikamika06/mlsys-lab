def check(workdir):
    from pte import plan
    import ref

    m = {"peak_val": 0.0, "peak_src": 0.0}
    raw = ref.get_raw_data()
    try:
        tensors = plan.parse_artifact(raw)
        peak, src = plan.find_peak(tensors)
        if peak == 2048:
            m["peak_val"] = 1.0
        if set(src) == {1, 2, 3}:
            m["peak_src"] = 1.0
    except Exception:
        pass
    return m
