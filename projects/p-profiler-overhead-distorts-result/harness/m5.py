import ref

def check(workdir):
    m = {"discrepancy_low": 0.0}
    env = ref.get_oracle_env(120.0)
    analyzer = ref.get_oracle_analyzer(env)
    if analyzer.check_discrepancy("sampling", 10.0) and not analyzer.check_discrepancy("instrumentation", 10.0):
        m["discrepancy_low"] = 1.0
    return m
