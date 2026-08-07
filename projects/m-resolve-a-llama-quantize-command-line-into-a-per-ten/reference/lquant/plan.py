def resolve_plan(model_tensors, default_type, overrides):
    plan = {}
    for name, shape in model_tensors.items():
        chosen = default_type
        for pat, ttype in overrides.items():
            if pat in name:
                chosen = ttype
                break
        plan[name] = chosen
    return plan
