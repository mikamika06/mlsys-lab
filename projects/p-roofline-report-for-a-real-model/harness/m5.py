def check(workdir):
    import ref
    from roofline.analysis import predict_total_time

    m = {"predict_matched": 0.0}
    hw = ref.get_test_hw()
    kernels = ref.get_test_kernels()

    try:
        expected = ref.ref_predict_total_time(hw, kernels)
        actual = predict_total_time(hw, kernels)
        if abs(expected - actual) < 1e-5:
            m["predict_matched"] = 1.0
    except NotImplementedError:
        pass
    except Exception:
        pass

    return m
