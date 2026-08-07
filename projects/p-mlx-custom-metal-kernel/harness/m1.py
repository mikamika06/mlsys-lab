def check(workdir):
    import ref
    m = {"baseline_measured": 0.0}
    try:
        from metal_op import kernel
        inputs = ref.get_test_inputs()
        res = kernel.measure_baseline(inputs[0])
        if res is not None:
            m["baseline_measured"] = 1.0
    except Exception:
        pass
    return m
