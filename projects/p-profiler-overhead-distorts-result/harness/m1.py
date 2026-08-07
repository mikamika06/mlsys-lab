import ref

def check(workdir):
    m = {"overhead_measured": 0.0}
    env = ref.get_oracle_env(120.0)
    try:
        val = env.measure("instrumentation")
        if val == 180.0:
            m["overhead_measured"] = 1.0
    except Exception:
        pass
    return m
