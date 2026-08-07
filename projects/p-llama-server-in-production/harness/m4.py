def check(workdir):
    from server.memory import check_memory_growth
    m = {"memory_bounded": 0.0}
    res = check_memory_growth([100, 50, -50, 20])
    if res["stable"] and res["peak"] == 150:
        m["memory_bounded"] = 1.0
    return m
