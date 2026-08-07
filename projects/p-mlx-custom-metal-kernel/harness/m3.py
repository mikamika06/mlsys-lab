def check(workdir):
    import ref
    m = {"boundary_correct": 0.0}
    try:
        from metal_op import kernel
        inputs = ref.get_test_inputs()
        ok = True
        for x in inputs:
            if not kernel.check_boundary(x):
                ok = False
        if ok:
            m["boundary_correct"] = 1.0
    except Exception:
        pass
    return m
