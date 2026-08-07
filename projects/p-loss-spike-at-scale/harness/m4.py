def check(workdir):
    import ref
    import numpy as np
    m = {"safe_is_correct": 0.0, "safe_is_stable": 0.0}
    try:
        from system import distributed, analysis
        tensors = ref.get_test_tensors()

        res = distributed.safe_all_reduce_sum(tensors)
        expected = sum(tensors)

        if np.allclose(res, expected, atol=1e-5):
            m["safe_is_correct"] = 1.0

        diff = analysis.check_determinism(distributed.safe_all_reduce_sum, tensors)
        if diff == 0.0:
            m["safe_is_stable"] = 1.0
    except Exception:
        pass
    return m
