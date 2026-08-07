import ref

def check(workdir):
    from runner.memory import analyze_duplicates

    m = {"no_duplicates": 0.0}
    procs = [{"pid": 100, "cmd": "llama"}, {"pid": 100, "cmd": "llama"}]
    res = analyze_duplicates(procs)
    oracle_dup = ref.get_oracle_duplicates(procs)
    if res.get("duplicate_detected") == oracle_dup:
        m["no_duplicates"] = 1.0
    return m
