import ref

def check(workdir):
    m = {"accounting_ok": 0.0}
    try:
        from qlora_fix.memory import account_memory
        res = account_memory(5000, 2000, 4000)
        expected = ref.oracle_account_memory(5000, 2000, 4000)
        if res.get("total_mb") == expected["total_mb"] and res.get("valid") == expected["valid"]:
            m["accounting_ok"] = 1.0
    except Exception:
        pass
    return m
