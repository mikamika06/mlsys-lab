import ref

def check(workdir):
    m = {"invariant_holds": 0.0}
    env = ref.get_oracle_env(120.0)
    analyzer = ref.get_oracle_analyzer(env)
    ok = analyzer.verify_invariant(lambda x: x > 100.0, ["clean", "sampling", "instrumentation"])
    if ok:
        m["invariant_holds"] = 1.0
    return m
