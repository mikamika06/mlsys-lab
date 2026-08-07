import ref

def check(workdir):
    m = {"under_budget": 0.0}
    try:
        from pte_plan.parser import parse_pte
        from pte_plan.planner import plan_buffers
        data = ref.generate_pte_artifact()
        parsed = parse_pte(data)
        planned_peak, _ = plan_buffers(parsed)
        budget = ref.get_device_budget(parsed)
        if planned_peak <= budget:
            m["under_budget"] = 1.0
    except Exception:
        pass
    return m
