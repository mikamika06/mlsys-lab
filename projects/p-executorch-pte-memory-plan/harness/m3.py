import ref

def check(workdir):
    m = {"separation_ok": 0.0}
    try:
        from pte_plan.parser import parse_pte
        from pte_plan.analyzer import separate_program_and_data
        data = ref.generate_pte_artifact()
        parsed = parse_pte(data)
        w_size, a_size = separate_program_and_data(parsed)
        ref_w, ref_a = ref.expected_separation(parsed)
        if w_size == ref_w and a_size == ref_a:
            m["separation_ok"] = 1.0
    except Exception:
        pass
    return m
