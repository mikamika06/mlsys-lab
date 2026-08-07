def check(workdir):
    import ref
    m = {"buggy_fails": 0.0, "stable_passes": 0.0}
    try:
        from system import analysis
        tensors = ref.get_test_tensors()

        diff_buggy = analysis.check_determinism(ref.buggy_reduce, tensors)
        if diff_buggy > 0.0:
            m["buggy_fails"] = 1.0

        def perfect_reduce(ts):
            return sum(ts)

        diff_perfect = analysis.check_determinism(perfect_reduce, tensors)
        if diff_perfect == 0.0:
            m["stable_passes"] = 1.0
    except Exception:
        pass
    return m
