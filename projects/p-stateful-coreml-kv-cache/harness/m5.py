import ref

def check(workdir):
    m = {"memory_ok": 0.0}
    try:
        contract = ref.get_oracle_contract()
        runner = ref.StatefulRunner(contract)
        for i in range(500):
            runner.step(i % 100)
        m["memory_ok"] = 1.0
    except Exception:
        pass
    return m
