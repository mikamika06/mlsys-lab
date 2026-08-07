def check(workdir):
    from runner import adapter

    m = {"all_tests_passed": 0.0}
    try:
        res = adapter.run_client_tests()
        if res is True:
            m["all_tests_passed"] = 1.0
    except Exception:
        pass
    return m
