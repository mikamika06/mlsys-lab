def check(workdir):
    import ref
    from roofline.analysis import optimization_potential

    m = {"potential_matched": 0.0}
    hw = ref.get_test_hw()
    kernels = ref.get_test_kernels()

    try:
        for k in kernels:
            expected = ref.ref_optimization_potential(hw, k)
            actual = optimization_potential(hw, k)
            if abs(expected - actual) > 1e-6:
                return m
        m["potential_matched"] = 1.0
    except NotImplementedError:
        pass
    except Exception:
        pass

    return m
