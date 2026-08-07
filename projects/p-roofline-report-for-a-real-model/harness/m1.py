def check(workdir):
    import ref
    from roofline.analysis import arithmetic_intensity

    m = {"intensity_matched": 0.0}
    cases = [(1000, 10), (0, 100), (5000, 0)]

    try:
        for flops, b in cases:
            expected = ref.ref_arithmetic_intensity(flops, b)
            actual = arithmetic_intensity(flops, b)
            if expected == float('inf'):
                if actual != float('inf'):
                    return m
            elif abs(expected - actual) > 1e-6:
                return m
        m["intensity_matched"] = 1.0
    except NotImplementedError:
        pass
    except Exception:
        pass

    return m
