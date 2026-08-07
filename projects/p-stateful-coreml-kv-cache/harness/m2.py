import ref

def check(workdir):
    m = {"export_ok": 0.0}
    try:
        pkg = ref.get_oracle_export()
        if isinstance(pkg, dict) and pkg.get("compiled") is True:
            m["export_ok"] = 1.0
    except Exception:
        pass
    return m
