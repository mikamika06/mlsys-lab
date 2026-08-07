import ref

def check(workdir):
    m = {"plan_reduced": 0.0}
    try:
        from pte_plan.parser import parse_pte
        from pte_plan.planner import plan_buffers
        data = ref.generate_pte_artifact()
        parsed = parse_pte(data)
        planned_peak, _ = plan_buffers(parsed)
        ref_planned, _ = ref.expected_plan(parsed)
        if planned_peak <= ref_planned:
            m["plan_reduced"] = 1.0
    except Exception:
        pass
    return m
