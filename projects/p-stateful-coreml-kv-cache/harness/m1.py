import ref

def check(workdir):
    m = {"contract_ok": 0.0}
    try:
        c = ref.define_state_contract(2, 4, 16, 64)
        if isinstance(c, dict) and "states" in c and len(c["states"]) == 4:
            m["contract_ok"] = 1.0
    except Exception:
        pass
    return m
