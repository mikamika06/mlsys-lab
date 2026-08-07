def check(workdir):
    import ref
    m = {"speedup_achieved": 0.0}
    try:
        from metal_op import kernel
        inputs = ref.get_test_inputs()
        val = kernel.measure_speedup(inputs[0])
        if val >= 1.0:
            m["speedup_achieved"] = float(val)
    except Exception:
        pass
    return m
