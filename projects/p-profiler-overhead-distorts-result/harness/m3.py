import ref

def check(workdir):
    m = {"mode_selected": 0.0}
    env = ref.get_oracle_env(120.0)
    analyzer = ref.get_oracle_analyzer(env)
    if analyzer.select_mode("macro") == "sampling" and analyzer.select_mode("micro") == "instrumentation":
        m["mode_selected"] = 1.0
    return m
