def check(workdir):
    import ref
    from roofline.analysis import kernel_performance

    m = {"perf_matched": 0.0}
    cases = [(1000000, 1.0), (5000000, 2.5), (100, 0.0)]

    try:
        for flops, t in cases:
            expected = ref.ref_kernel_performance(flops, t)
            actual = kernel_performance(flops, t)
            if abs(expected - actual) > 1e-6:
                return m
        m["perf_matched"] = 1.0
    except NotImplementedError:
        pass
    except Exception:
        pass

    return m
