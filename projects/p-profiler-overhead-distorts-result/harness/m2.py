import ref

def check(workdir):
    m = {"comparison_ok": 0.0}
    env = ref.get_oracle_env(120.0)
    analyzer = ref.get_oracle_analyzer(env)
    res = analyzer.compare_modes("sampling", "instrumentation")
    if "diff" in res and res["diff"] == 55.0:
        m["comparison_ok"] = 1.0
    return m
