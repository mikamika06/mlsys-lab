def expand_target_modules(model_structure, target_modules="all-linear"):
    modules = model_structure.get("modules", {})
    if target_modules != "all-linear":
        if isinstance(target_modules, str):
            return [target_modules]
        return list(target_modules)
    expanded = []
    lm_head_name = model_structure.get("lm_head_name", "lm_head")
    for name, spec in modules.items():
        if spec.get("type") == "linear":
            if name == lm_head_name or name.endswith("." + lm_head_name):
                continue
            expanded.append(name)
    return sorted(expanded)
