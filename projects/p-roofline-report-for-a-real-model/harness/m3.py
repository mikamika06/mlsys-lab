def check(workdir):
    import ref
    from roofline.analysis import roofline_ceiling

    m = {"ceiling_matched": 0.0}
    hw = ref.get_test_hw()
    cases = [0.1, 10.0, 1000.0, float('inf')]

    try:
        for intensity in cases:
            expected = ref.ref_roofline_ceiling(hw, intensity)
            actual = roofline_ceiling(hw, intensity)
            if abs(expected - actual) > 1e-6:
                return m
        m["ceiling_matched"] = 1.0
    except NotImplementedError:
        pass
    except Exception:
        pass

    return m
