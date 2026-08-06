def diagnose_wrap_policy(module_list, config):
    misconfigured = []
    min_size = config.get("min_size", 1024)
    for mod in module_list:
        name = mod.get("name", "")
        size = mod.get("size", 0)
        wrapped = mod.get("wrapped", False)
        if size >= min_size and not wrapped:
            misconfigured.append(name)
        elif size < min_size and wrapped:
            misconfigured.append(name)
    return {
        "is_valid": len(misconfigured) == 0,
        "misconfigured_modules": sorted(misconfigured)
    }
