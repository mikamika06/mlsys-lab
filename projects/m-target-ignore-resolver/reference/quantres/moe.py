def build_moe_ignore_list(model_structure):
    ignores = []
    for name in model_structure.get("modules", []):
        if "gate" in name or "router" in name or "norm" in name:
            ignores.append(name)
    return sorted(list(set(ignores)))
