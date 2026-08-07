import ref

def check(workdir):
    m = {"parse_ok": 0.0, "tensors_count": 0.0}
    try:
        from pte_plan.parser import parse_pte
        data = ref.generate_pte_artifact()
        parsed = parse_pte(data)
        if parsed and "tensors" in parsed:
            m["parse_ok"] = 1.0
            m["tensors_count"] = float(len(parsed["tensors"]))
    except Exception:
        pass
    return m
