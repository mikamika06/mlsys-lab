import ref

def check(workdir):
    from app.pipeline import run_1000_times
    results = run_1000_times()

    m = {"count_ok": 1.0 if len(results) == 1000 else 0.0, "all_valid": 1.0}
    for res in results:
        if len(res) != 9 or res[0] != 0 or res[-1] != 1:
            m["all_valid"] = 0.0
            break
    return m
