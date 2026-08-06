import ref

def check(workdir):
    from specfail import classifier
    scenarios = ref.get_scenarios()
    want = ref.classify_scenarios(scenarios)
    try:
        got = classifier.classify_scenarios(scenarios)
    except Exception as e:
        return {"scenarios_classified": 0.0, "_note": f"exception in classify_scenarios: {e}"}
    
    ok = 0
    if isinstance(got, list) and len(got) == len(want):
        for g_item, w_item in zip(got, want):
            if g_item.get("category") == w_item.get("category"):
                ok += 1
    return {"scenarios_classified": float(ok)}
