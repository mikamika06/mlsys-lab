def check(workdir):
    import ref
    m = {"kernel_compiled": 0.0}
    try:
        from metal_op import kernel
        inputs = ref.get_test_inputs()
        res = kernel.run_custom_kernel(inputs[0])
        if res is not None:
            m["kernel_compiled"] = 1.0
    except Exception:
        pass
    return m
